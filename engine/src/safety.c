/* safety.c - gain ramps, the peak limiter, the soft clipper, default params and status strings
 * (20-engine 11, contract 4 "DSP chain blocks" and 4.2 "Safety chain").
 *
 * Package E-0 wrote the primitives. E-2 added the limiter's min-gain report, the runaway guard and the
 * cue generator at the end of the file.
 * Threads: every function is RT-safe (no allocation, no locks, bounded loops) and may run on any thread,
 * but one es_ramp or es_limiter must only be used by one thread at a time.
 */
#include <math.h>

#include "internal.h"

/* ------------------------------------------------------------------------------------------------
 * es_ramp - a gain that moves from `start` to `target` over a fixed time.
 *   phase runs 0..1 in `step` increments per sample;
 *   shape 0 (linear):          w = phase
 *   shape 1 (raised cosine):   w = 0.5 - 0.5 cos(pi * phase)   (slow start, slow end)
 *   gain = start + (target - start) * w
 * A new es_ramp_to always starts from the current gain, so a ramp that is interrupted never jumps.
 * ---------------------------------------------------------------------------------------------- */
void es_ramp_init(es_ramp *r, float sr, float gain) {
  r->gain = gain;
  r->start = gain;
  r->target = gain;
  r->step = 0.0f;
  r->phase = 1.0f;     /* finished: nothing armed */
  r->sr = sr;
  r->shape = 0;
}

void es_ramp_to(es_ramp *r, float target, float ms, int shape) {
  r->target = target;
  r->shape = shape;
  if (!(ms > 0.0f)) {  /* ms <= 0 (or NaN) jumps straight to the target */
    r->gain = target;
    r->start = target;
    r->phase = 1.0f;
    r->step = 0.0f;
    return;
  }
  r->start = r->gain;  /* continue from wherever the gain is now */
  r->phase = 0.0f;
  r->step = 1000.0f / (ms * r->sr);
}

void es_ramp_process(es_ramp *r, float *buf, int n) {
  for (int i = 0; i < n; i++) {
    if (r->phase < 1.0f) {
      r->phase = fminf(r->phase + r->step, 1.0f);
      float w = r->shape == 1 ? 0.5f - 0.5f * cosf((float)ES_PI * r->phase) : r->phase;
      if (r->phase >= 1.0f) r->gain = r->target;   /* land exactly on the target (no rounding left) */
      else                  r->gain = r->start + (r->target - r->start) * w;
    }
    buf[i] *= r->gain;
  }
}

/* ------------------------------------------------------------------------------------------------
 * Peak limiter with zero lookahead (so it adds no latency). Instant attack: when a sample is louder
 * than the ceiling, the gain drops at once to ceiling / |x|. Release: the gain moves back toward the
 * gain the current sample allows with a one-pole filter (63% of the way in release_ms), never above
 * it. So |output| <= ceiling for every sample (plus one float rounding step).
 * ---------------------------------------------------------------------------------------------- */
void es_limiter_init(es_limiter *l, float sr, float ceiling_dbfs, float release_ms) {
  l->ceiling = powf(10.0f, ceiling_dbfs / 20.0f);           /* -3 dBFS = 0.7079458 */
  l->rel_coef = expf(-1.0f / (release_ms * 0.001f * sr));   /* 48 kHz, 50 ms: 0.9995834 */
  l->gain = 1.0f;
  l->min_gain = 1e-6f;
}

void es_limiter_process(es_limiter *l, float *x, int n) {
  (void)es_limiter_process_min(l, x, n);                     /* the loop lives below (E-2) */
}

/* ------------------------------------------------------------------------------------------------
 * Soft clip with its knee at -1 dBFS (k = 0.8912509). Below the knee the signal passes unchanged.
 * Above it, k + r * tanh((|x| - k) / r) with r = 1 - k: continuous with slope 1 at the knee and never
 * beyond 1 in magnitude. After the -3 dBFS limiter it only acts if the ceiling were ever raised.
 * ---------------------------------------------------------------------------------------------- */
void es_softclip(float *buf, int n) {
  const float k = 0.8912509f, r = 1.0f - 0.8912509f;
  for (int i = 0; i < n; i++) {
    float a = fabsf(buf[i]);
    if (a > k) buf[i] = copysignf(k + r * tanhf((a - k) / r), buf[i]);
  }
}

/* ------------------------------------------------------------------------------------------------
 * Defaults and status strings (contract 4).
 * ---------------------------------------------------------------------------------------------- */
void es_params_default(es_params *p) {
  p->voice_db = 0.0f;
  p->room_db = -60.0f;
  p->max_gain_db = 18.0f;
  p->out_gain_db = 0.0f;
  p->slot_gain[0] = 1.0f;
  for (int s = 1; s < ES_MAX_INSTANCES; s++) p->slot_gain[s] = 0.0f;
  p->xfade_ms = 150.0f;
  p->duck_own = 1;
  p->gate = 1;
  p->agc = 1;
  p->bypass = 0;
}

const char *es_status_str(es_status s) {
  switch (s) {
    case ES_OK:               return "ok";
    case ES_ERR_ARG:          return "invalid argument";
    case ES_ERR_FORMAT:       return "bad file format";
    case ES_ERR_VERSION:      return "unsupported version";
    case ES_ERR_CONFIG:       return "model config does not match the engine";
    case ES_ERR_NOMEM:        return "out of memory";
    case ES_ERR_ENROLL_SHORT: return "not enough speech in the enrollment clip";
    case ES_ERR_BUSY:         return "busy, try again";
    case ES_ERR_STATE:        return "wrong state";
    case ES_ERR_RATE:         return "unsupported sample rate";
    case ES_ERR_PROBE_WEAK:   return "latency probe signal too weak";
    case ES_ERR_SELFTEST:     return "self-test failed";
    case ES_ERR_THREAD:       return "worker thread setup failed";
  }
  return "unknown status";
}

/* ------------------------------------------------------------------------------------------------
 * Package E-2 additions (20-engine 11): limiter gain report, runaway guard, cue beeps.
 * All render-thread, RT-safe.
 * ---------------------------------------------------------------------------------------------- */

/* The limiter loop of es_limiter_process, also returning the smallest gain it applied (1 if it never
   reduced anything). The render reports -20 log10(min gain) as limiter_gr_db. */
float es_limiter_process_min(es_limiter *l, float *x, int n) {
  float min_gain = 1.0f;
  for (int i = 0; i < n; i++) {
    float a = fabsf(x[i]);
    float gt = a > l->ceiling ? l->ceiling / a : 1.0f;      /* the most gain this sample allows */
    if (gt < l->gain) l->gain = gt;                          /* instant attack */
    else l->gain = gt + (l->gain - gt) * l->rel_coef;        /* release toward gt, never above it */
    if (l->gain < l->min_gain) l->gain = l->min_gain;
    if (l->gain < min_gain) min_gain = l->gain;
    x[i] *= l->gain;
  }
  return min_gain;
}

/* Runaway guard. Per call (one render chunk of n samples):
     a = 1 - exp(-n / (0.05 sr))                    one-pole step for a 50 ms time constant
     in_ms  += a * (mean(xin^2) - in_ms)            sanitized mic input
     out_ms += a * (mean(y^2)   - out_ms)           output after the trim (stage 3), before the duck
   Condition: out_ms > 1000 * max(in_ms, 1e-8) (30 dB louder than the mic) and out_ms > 1e-6 (-60 dBFS).
   Condition true for 0.2 s in a row -> howl = 1, duck to -20 dB over 5 ms (linear).
   Condition false for 2 s in a row while howling -> howl = 0, back to 0 dB over 400 ms (raised cosine).
   The measurement is taken before the duck, so the duck never releases itself. */
void es_guard_init(es_guard *g, float sr) {
  g->sr = sr;
  g->in_ms = 0.0f;
  g->out_ms = 0.0f;
  g->over = 0;
  g->under = 0;
  g->howl = 0;
  es_ramp_init(&g->duck, sr, 1.0f);
}

int es_guard_process(es_guard *g, const float *xin, float *y, int n) {
  if (n <= 0) return g->howl;
  float si = 0.0f, so = 0.0f;
  for (int i = 0; i < n; i++) {
    si += xin[i] * xin[i];
    so += y[i] * y[i];
  }
  float a = 1.0f - expf(-(float)n / (0.05f * g->sr));
  g->in_ms += a * (si / (float)n - g->in_ms);
  g->out_ms += a * (so / (float)n - g->out_ms);

  int cond = g->out_ms > 1000.0f * fmaxf(g->in_ms, 1e-8f) && g->out_ms > 1e-6f;
  if (cond) {
    g->over += n;
    g->under = 0;
    if (!g->howl && g->over >= (int64_t)(0.2f * g->sr)) {
      g->howl = 1;
      es_ramp_to(&g->duck, 0.1f, 5.0f, 0);          /* -20 dB */
    }
  } else {
    g->over = 0;
    if (g->howl) {
      g->under += n;
      if (g->under >= (int64_t)(2.0f * g->sr)) {
        g->howl = 0;
        g->under = 0;
        es_ramp_to(&g->duck, 1.0f, 400.0f, 1);
      }
    }
  }
  es_ramp_process(&g->duck, y, n);
  return g->howl;
}

/* Cue beeps. The sequence of b beeps is b * 150 ms long minus the last 90 ms gap:
     beep k occupies [150k, 150k + 60) ms; inside it the tone is 0.1258925 sin(phase) (-18 dBFS peak)
     times a 5 ms raised-cosine fade at each end; the gaps are silence.
   The phase advances by 2 pi 880 / sr every sample of the sequence and wraps at 2 pi. */
void es_cue_init(es_cue *c, float sr) {
  c->sr = sr;
  c->phase = 0.0f;
  c->dphase = (float)(2.0 * ES_PI * 880.0 / sr);
  c->beep_len = (int)(0.060f * sr + 0.5f);
  c->edge_len = (int)(0.005f * sr + 0.5f);
  c->period_len = (int)(0.150f * sr + 0.5f);
  c->total = 0;
  c->pos = 0;
}

void es_cue_start(es_cue *c, int beeps) {
  if (beeps < 1) beeps = 1;
  if (beeps > 3) beeps = 3;
  c->total = (beeps - 1) * c->period_len + c->beep_len;
  c->pos = 0;
  c->phase = 0.0f;
}

void es_cue_add(es_cue *c, float *y, int n) {
  const float two_pi = (float)(2.0 * ES_PI);
  for (int i = 0; i < n && c->pos < c->total; i++, c->pos++) {
    int t = c->pos % c->period_len;                  /* position inside this beep + gap */
    if (t < c->beep_len) {
      float env = 1.0f;
      if (t < c->edge_len) env = 0.5f - 0.5f * cosf((float)ES_PI * (float)t / (float)c->edge_len);
      else if (t >= c->beep_len - c->edge_len)
        env = 0.5f - 0.5f * cosf((float)ES_PI * (float)(c->beep_len - 1 - t) / (float)c->edge_len);
      y[i] += 0.1258925f * env * sinf(c->phase);
    }
    c->phase += c->dphase;
    if (c->phase >= two_pi) c->phase -= two_pi;
  }
  if (c->pos >= c->total) c->total = 0;              /* finished: idle until the next request */
}
