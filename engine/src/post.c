/* post.c - post-processing of one 64-sample hop at 16 kHz (20-engine 10, contract 4.2
 * "Post-processing per hop at 16 kHz").
 *
 * Thread: the real-time worker only (called by es_engine_run_hops in pipeline.c). RT-safe: no
 * allocation, no locks, fixed-length loops; expf / powf / log10f / sqrtf are plain libm calls.
 *
 * The seven steps, in order (each one is a numbered block in es_post_hop below):
 *   (1) activity gate per slot   0 dB at p_target >= 0.5 down to -12 dB at p_target <= 0.2;
 *                                opens within 1 hop, closes with a 200 ms time constant
 *   (2) crossfade sum            slot powers move toward slot_gain^2 by at most 4 / xfade_ms per hop
 *                                (equal power during a switch), sum = sum_s sqrt(P_s) * gate_s * ys[s]
 *   (3) own-voice duck           -15 dB while max p_self > 0.5 (if duck_own), 1-hop attack, 150 ms release
 *   (4) slow AGC                 RMS of the sum over 400 ms, only while max p_target > 0.5, target
 *                                -26 dBFS + voice_db, at most 0.024 dB per hop (6 dB/s), within
 *                                [-12, max_gain_db]; frozen otherwise; agc off slews back to 0 dB
 *   (5) gain cap                 50 ms mean squares of the output of (4) and of x_d; any excess of their
 *                                ratio over +24 dB is removed at once and released with 200 ms (exactly
 *                                the 20-engine 10 step 5 formula, see the note at step 5)
 *   (6) room mix-back            + 10^(room_db / 20) * x_d, forced to 0 at once while own voice is
 *                                detected (whatever duck_own says), recovering with 150 ms
 *   (7) bypass                   the mix moves toward `bypass` by 0.2 per hop; at 1 the output is x_d
 *
 * How gains are applied: each gain gets one new value per hop. Inside the hop it ramps linearly
 * sample by sample from the previous hop's value to the new one, so there is no zipper noise. The gain
 * cap and the room envelope are the exceptions: when they fall, the new lower value is used for the
 * whole hop at once (the contract says "removed at once" and "forced to 0 at once").
 */
#include <math.h>
#include <string.h>

#include "internal.h"

static float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
static float db_to_lin(float db) { return powf(10.0f, db / 20.0f); }

static float mean_sq(const float *x) {
  float s = 0.0f;
  for (int i = 0; i < ES_HOP; i++) s += x[i] * x[i];
  return s / (float)ES_HOP;
}

/* dst[i] += src[i] * g(i) with g ramping linearly from g0 (exclusive) to g1 (reached at i = 63). */
static void ramp_mac(float *dst, const float *src, float g0, float g1) {
  float dg = (g1 - g0) / (float)ES_HOP;
  for (int i = 0; i < ES_HOP; i++) dst[i] += src[i] * (g0 + dg * (float)(i + 1));
}

/* dst[i] = src[i] * g(i), same ramp. dst may equal src. */
static void ramp_mul(float *dst, const float *src, float g0, float g1) {
  float dg = (g1 - g0) / (float)ES_HOP;
  for (int i = 0; i < ES_HOP; i++) dst[i] = src[i] * (g0 + dg * (float)(i + 1));
}

/* Room gain for a room_db setting: -60 dB (anything at or below -59.5) means off. */
static float room_gain(const es_params *prm) {
  return prm->room_db <= -59.5f ? 0.0f : db_to_lin(prm->room_db);
}

void es_post_init(es_post *p, const es_params *prm) {
  memset(p, 0, sizeof(*p));
  p->a200 = 1.0f - expf(-4.0f / 200.0f);    /* 0.0198013 */
  p->a150 = 1.0f - expf(-4.0f / 150.0f);    /* 0.0263143 */
  p->a400 = 1.0f - expf(-4.0f / 400.0f);    /* 0.0099502 */
  p->a50 = 1.0f - expf(-4.0f / 50.0f);      /* 0.0768837 */
  for (int s = 0; s < ES_MAX_INSTANCES; s++) {
    float g = clampf(prm->slot_gain[s], 0.0f, 1.0f);
    p->g_db[s] = 0.0f;                      /* gates start open */
    p->pw[s] = g * g;                       /* slots start at their configured power */
    p->slot_lin[s] = g;
  }
  p->d_db = 0.0f;
  p->agc_db = 0.0f;
  p->agc_primed = 0;
  p->main_lin = 1.0f;
  p->cap_db = 0.0f;
  p->cap_lin = 1.0f;
  p->r_env = 1.0f;
  p->room_lin = room_gain(prm);
  p->bm = prm->bypass ? 1.0f : 0.0f;
}

void es_post_hop(es_post *p, const es_params *prm, int n_slots, const float (*ys)[ES_HOP],
                 const float (*hd)[2], const int *active, const float *xd, float *y,
                 float *pt_max_out, float *ps_max_out) {
  /* Maxima over the active slots (0 if none). */
  float pt_max = 0.0f, ps_max = 0.0f;
  for (int s = 0; s < n_slots; s++) {
    if (!active[s]) continue;
    pt_max = fmaxf(pt_max, hd[s][0]);
    ps_max = fmaxf(ps_max, hd[s][1]);
  }
  int own = ps_max > 0.5f;

  /* (1) gate and (2) crossfade sum. */
  float pstep = 4.0f / clampf(prm->xfade_ms, 4.0f, 2000.0f);   /* most power change per hop */
  memset(p->sum, 0, sizeof(p->sum));
  for (int s = 0; s < n_slots; s++) {
    float pt = active[s] ? hd[s][0] : 0.0f;
    float gt = pt >= 0.5f ? 0.0f : (pt <= 0.2f ? -12.0f : -12.0f * (0.5f - pt) / 0.3f);
    if (!prm->gate) p->g_db[s] = 0.0f;
    else if (gt > p->g_db[s]) p->g_db[s] = gt;                  /* opens within this hop */
    else p->g_db[s] += p->a200 * (gt - p->g_db[s]);             /* closes with 200 ms */

    float sg = clampf(prm->slot_gain[s], 0.0f, 1.0f);
    p->pw[s] += clampf(sg * sg - p->pw[s], -pstep, pstep);       /* linear power ramp */
    float g = sqrtf(p->pw[s]) * db_to_lin(p->g_db[s]);
    ramp_mac(p->sum, ys[s], p->slot_lin[s], g);
    p->slot_lin[s] = g;
  }

  /* (3) own-voice duck: lower target jumps (1-hop attack), higher target releases with 150 ms. */
  float dt = (prm->duck_own && own) ? -15.0f : 0.0f;
  if (dt < p->d_db) p->d_db = dt;
  else p->d_db += p->a150 * (dt - p->d_db);

  /* (4) slow AGC, measured on the sum of (2), before the duck, so ducking never pumps it. */
  if (prm->agc) {
    if (pt_max > 0.5f) {
      float ms = mean_sq(p->sum);
      if (!p->agc_primed) {
        p->agc_ms = ms;
        p->agc_primed = 1;
      } else {
        p->agc_ms += p->a400 * (ms - p->agc_ms);
      }
      float want = clampf(-26.0f + prm->voice_db - 10.0f * log10f(p->agc_ms + 1e-12f), -12.0f, prm->max_gain_db);
      p->agc_db += clampf(want - p->agc_db, -0.024f, 0.024f);
    }
  } else {
    p->agc_db += clampf(0.0f - p->agc_db, -0.024f, 0.024f);    /* off: back to 0 dB at 6 dB/s */
  }
  p->agc_db = clampf(p->agc_db, -12.0f, prm->max_gain_db);      /* a lowered ceiling applies at once */

  /* (5) gain cap. ypre = sum * duck * AGC is the output of (4). */
  float main = db_to_lin(p->d_db + p->agc_db);
  ramp_mul(p->ypre, p->sum, p->main_lin, main);
  p->main_lin = main;
  p->in_ms += p->a50 * (mean_sq(xd) - p->in_ms);
  p->out_ms += p->a50 * (mean_sq(p->ypre) - p->out_ms);
  float excess = 10.0f * log10f(p->out_ms + 1e-10f) - 10.0f * log10f(p->in_ms + 1e-10f) - 24.0f;
  float ct = -fmaxf(excess, 0.0f);
  if (ct < p->cap_db) p->cap_db = ct;                           /* excess removed at once */
  else p->cap_db += p->a200 * (ct - p->cap_db);                 /* released with 200 ms */
  float cap = db_to_lin(p->cap_db);
  /* Note (E-2 fix pass, limits from DECISIONS L 2026-11-08). The cap looks at 50 ms averages, so when
     the gain in front of it rises faster than that (the 3 deep-filter taps building up at a tone onset),
     the capped output runs over +24 dB for a short while: with every tap at 1.5 + 1.5j test_latency
     measures 25.05 dB in the worst 50 ms window at the onset and 24.0 dB once settled. The accepted
     limits are 24.1 dB in steady state and 25.6 dB in the first 200 ms after an onset. An extra
     per-hop bound on the capped output cut most of that overshoot but pumped the gain by up to 1.6 dB
     from hop to hop on steady noise, so it was taken out. The -3 dBFS limiter and soft clip still
     bound every peak (the hearing-safety bound). */
  ramp_mul(y, p->ypre, fminf(p->cap_lin, cap), cap);            /* a falling cap does not ramp */
  p->cap_lin = cap;

  /* (6) room mix-back. The envelope drops to 0 at once on own voice (independent of duck_own: the
     delayed room path must never replay the user's voice) and recovers with 150 ms. */
  if (own) p->r_env = 0.0f;
  else p->r_env += p->a150 * (1.0f - p->r_env);
  float rg = room_gain(prm) * p->r_env;
  if (rg > 0.0f || p->room_lin > 0.0f) ramp_mac(y, xd, fminf(p->room_lin, rg), rg);
  p->room_lin = rg;

  /* (7) bypass: y = (1 - bm) y + bm xd, bm ramps by 0.2 per hop. At bm = 1 the output is exactly x_d. */
  float bm_prev = p->bm;
  float bt = prm->bypass ? 1.0f : 0.0f;
  p->bm += clampf(bt - p->bm, -0.2f, 0.2f);
  if (p->bm >= 1.0f && bm_prev >= 1.0f) {
    memcpy(y, xd, sizeof(float) * ES_HOP);
  } else if (p->bm > 0.0f || bm_prev > 0.0f) {
    float db = (p->bm - bm_prev) / (float)ES_HOP;
    for (int i = 0; i < ES_HOP; i++) {
      float b = bm_prev + db * (float)(i + 1);
      y[i] = (1.0f - b) * y[i] + b * xd[i];
    }
  }

  *pt_max_out = pt_max;
  *ps_max_out = ps_max;
}
