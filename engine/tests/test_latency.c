/* test_latency.c - engine-level tests with synthetic models (20-engine 14.5, plus the 14.1 rows that
 * need a whole engine: alignment, gain cap and the conditioning hand-off race). Needs no ML export.
 *
 * Threads: single-threaded (this thread is both the audio thread and the worker: render, then
 * es_engine_run_hops, like the 9.4 prefill simulation), except the hand-off row, which runs a writer
 * thread against a render + run_hops thread.
 *
 * Rows
 *   impulse_bypass_RATE_FRAMES_SLACK  bypass engine: delay of an impulse (mean group delay 1 to 4 kHz)
 *                                     equals es_engine_latency_ms within one 16 kHz sample, 0 underruns
 *   impulse_model_RATE_FRAMES_SLACK   the same with the synthetic all-pass model and bypass off
 *   model_matches_bypass_*            the all-pass model path gives the bypass output sample by sample
 *   alignment_model_output_equals_xd  raw model output of hop j equals the input of hop j - 1 (8.6)
 *   gain_cap_*                        every coefficient 1.5 + 1.5j, AGC at its maximum: once settled
 *                                     (200 ms into the tone) the 50 ms output RMS stays <= input RMS +
 *                                     24.1 dB; in the first 200 ms after the onset <= 25.6 dB
 *                                     (DECISIONS L 2026-11-08)
 *   handoff_reset_kept_while_stopped  a reset asked for while the worker is stopped survives a second
 *                                     set_cond before the worker restarts
 *   underrun_once_per_callback        a starved callback split into chunks counts one underrun
 *   handoff_*                         a writer thread calls set_cond 1,000,000 times against hops
 *   slack_step_*                      two underruns raise slack 43 -> 64 (+1.3125 ms), the inserted
 *                                     zeros carry 1 ms ramps (no step above the ramp slope)
 * Every check prints "PASS name value limit" or "FAIL name value limit"; exit 1 on any failure.
 */
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

#include "internal.h"
#include "ring.h"

/* ---------------------------------------------------------------------------------------------- */
/* Harness                                                                                         */
/* ---------------------------------------------------------------------------------------------- */
static int g_pass = 0, g_fail = 0;

static void check(const char *name, int ok, double value, double limit) {
  printf("%s %s %.6g %.6g\n", ok ? "PASS" : "FAIL", name, value, limit);
  if (ok) g_pass++; else g_fail++;
}
static void check_le(const char *name, double value, double limit) { check(name, value <= limit, value, limit); }
static void check_true(const char *name, int ok) { check(name, ok, ok ? 1 : 0, 1); }

static uint32_t g_rng = 2463534242u;
static uint32_t xs32(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }
static float urand(void) { return (float)((xs32() >> 8) * (1.0 / 16777216.0)) * 2.0f - 1.0f; }

/* ---------------------------------------------------------------------------------------------- */
/* Engine helpers                                                                                  */
/* ---------------------------------------------------------------------------------------------- */
/* A 1-instance engine that this thread drives (manual worker), unmuted. */
static es_engine *make_engine(const es_model *m, int rate, int frames, int slack) {
  es_engine_config cfg = { rate, frames, 4096, 1, slack };
  es_engine *e = NULL;
  if (es_engine_new(m, &cfg, &e) != ES_OK) return NULL;
  es_engine_set_manual_worker(e, 1);
  es_engine_set_mute(e, 0);
  return e;
}

/* Params with gate, duck, AGC and room off (the 14.5 setup). */
static void plain_params(es_engine *e, int bypass) {
  es_params p;
  es_params_default(&p);
  p.gate = 0;
  p.duck_own = 0;
  p.agc = 0;
  p.room_db = -60;
  p.bypass = bypass;
  es_engine_set_params(e, &p);
}

/* Fill slot 0 with a valid conditioning set (the synthetic models ignore its values). */
static void fill_slot0(es_engine *e) {
  es_cond *c = es_cond_new();
  c->n_tokens = 4;
  es_engine_set_cond(e, 0, c, 1);
  es_cond_free(c);
}

/* One callback: render `n` frames, then run the hops it made (an instant worker). */
static void drive(es_engine *e, const float *in, float *out, int n) {
  es_engine_render(e, in, out, n);
  es_engine_run_hops(e);
}

/* Run a whole input buffer through the engine in callbacks of `frames`. */
static void run_buffer(es_engine *e, const float *in, float *out, int total, int frames) {
  for (int off = 0; off < total; off += frames) {
    int n = total - off < frames ? total - off : frames;
    drive(e, in + off, out + off, n);
  }
}

static uint64_t underruns(const es_engine *e) {
  es_stats s;
  es_engine_get_stats(e, &s);
  return s.underruns;
}

/* Mean group delay in ms over 1 to 4 kHz of the 8,192 samples y[0..8192) at rate sr: DFT (double) at
   each bin, unwrap the phase, delay = -(phase(f2) - phase(f1)) / (2 pi (f2 - f1)), which is the mean
   of the per-bin group delays over the band. */
static double group_delay_ms(const float *y, double sr) {
  enum { N = 8192 };
  static double ctab[N], stab[N];                 /* cos and sin of 2 pi m / N */
  if (ctab[0] == 0.0)
    for (int m = 0; m < N; m++) { ctab[m] = cos(2.0 * ES_PI * m / N); stab[m] = sin(2.0 * ES_PI * m / N); }
  int k1 = (int)ceil(1000.0 * N / sr), k2 = (int)floor(4000.0 * N / sr);
  double prev = 0, unwrapped = 0, first = 0;
  for (int k = k1; k <= k2; k++) {
    double re = 0, im = 0;
    for (int n = 0; n < N; n++) {                 /* X[k] = sum y[n] exp(-2 pi j k n / N) */
      int m = (int)(((int64_t)k * n) % N);
      re += y[n] * ctab[m];
      im -= y[n] * stab[m];
    }
    double ph = atan2(im, re);
    if (k == k1) {
      unwrapped = first = ph;
    } else {
      double d = ph - prev;
      while (d > ES_PI) d -= 2.0 * ES_PI;
      while (d < -ES_PI) d += 2.0 * ES_PI;
      unwrapped += d;
    }
    prev = ph;
  }
  double f1 = k1 * sr / N, f2 = k2 * sr / N;
  return -(unwrapped - first) / (2.0 * ES_PI * (f2 - f1)) * 1000.0;
}

/* ---------------------------------------------------------------------------------------------- */
/* 14.5 impulse delay: bypass and the all-pass model, every rate, callback size and slack          */
/* ---------------------------------------------------------------------------------------------- */
/* 0.6 s of silence (past the 400 ms fade-in), an impulse of 0.5, 0.5 s of silence. Returns the
   delay estimate in ms; writes the output (total samples) and the underrun count. */
static double impulse_run(const es_model *m, int rate, int frames, int slack, int use_model,
                          float *out, int *total_out, uint64_t *ur, float *latency) {
  es_engine *e = make_engine(m, rate, frames, slack);
  plain_params(e, !use_model);
  if (use_model) fill_slot0(e);
  int i0 = (int)(0.6 * rate), total = i0 + (int)(0.5 * rate);
  float *in = calloc((size_t)total, sizeof(float));
  in[i0] = 0.5f;
  run_buffer(e, in, out, total, frames);
  *total_out = total;
  *ur = underruns(e);
  *latency = es_engine_latency_ms(e);
  double est = group_delay_ms(out + i0, rate);
  free(in);
  es_engine_free(e);
  return est;
}

static void test_impulse(const es_model *ap) {
  static const int rates[2] = { 48000, 44100 };
  static const int frames[8] = { 16, 32, 64, 128, 192, 256, 480, 1104 };
  static const int slacks[3] = { 22, 43, 64 };
  float *ob = malloc(sizeof(float) * 60000), *om = malloc(sizeof(float) * 60000);
  double worst_match = 0;
  for (int r = 0; r < 2; r++)
    for (int f = 0; f < 8; f++)
      for (int s = 0; s < 3; s++) {
        char name[96];
        int tb, tm;
        uint64_t ub, um;
        float lb, lm;
        double eb = impulse_run(ap, rates[r], frames[f], slacks[s], 0, ob, &tb, &ub, &lb);
        double em = impulse_run(ap, rates[r], frames[f], slacks[s], 1, om, &tm, &um, &lm);
        snprintf(name, sizeof(name), "impulse_bypass_%d_%d_%d", rates[r], frames[f], slacks[s]);
        check(name, fabs(eb - lb) <= 0.0625 && ub == 0, fabs(eb - lb), 0.0625);
        snprintf(name, sizeof(name), "impulse_model_%d_%d_%d", rates[r], frames[f], slacks[s]);
        check(name, fabs(em - lm) <= 0.0625 && um == 0, fabs(em - lm), 0.0625);
        for (int i = 0; i < tb && i < tm; i++) worst_match = fmax(worst_match, fabs(ob[i] - om[i]));
        if (r == 0 && f == 2 && s == 1)
          printf("INFO 48 kHz / 64 frames / slack 43: latency_ms %.4f, bypass estimate %.4f, model %.4f\n",
                 lb, eb, em);
      }
  check_le("model_matches_bypass_all_48_configs_max_abs", worst_match, 1e-5);
  free(ob);
  free(om);
}

/* ---------------------------------------------------------------------------------------------- */
/* Alignment (8.6): the all-pass model's raw output of hop j equals the input of hop j - 1         */
/* ---------------------------------------------------------------------------------------------- */
static void test_alignment(const es_model *ap) {
  es_engine *e = make_engine(ap, 48000, 64, 43);
  plain_params(e, 0);
  fill_slot0(e);
  enum { TOTAL = 48000 };                          /* 1 s of noise: 250 hops */
  float *in = malloc(sizeof(float) * TOTAL), *out = malloc(sizeof(float) * TOTAL);
  for (int i = 0; i < TOTAL; i++) in[i] = 0.5f * urand();
  run_buffer(e, in, out, TOTAL, 64);
  enum { H = 16000 };
  static float xin[H], yout[H];
  uint64_t c1 = 0, c2 = 0;
  int nx = hist_read(&e->inhist, &c1, xin, H);
  int ny = hist_read(&e->outhist[0], &c2, yout, H);
  double err = 0;
  for (int n = 128; n < ny && n < nx; n++) err = fmax(err, fabs(yout[n] - xin[n - 64]));
  check_true("alignment_hops_cover_the_input", ny >= 15936 && nx >= ny);
  check_le("alignment_model_output_equals_xd_after_hop2", err, 1e-6);
  free(in);
  free(out);
  es_engine_free(e);
}

/* ---------------------------------------------------------------------------------------------- */
/* Gain cap (contract 4.2 post step 5): every deep-filter coefficient 1.5 + 1.5j and the AGC at its */
/* maximum. A 1 kHz tone (a multiple of 250 Hz, so the 3 taps add in phase every 4 ms hop) gets     */
/* 3 x |1.5 + 1.5j| = 6.36x (+16.1 dB) from the filter and +18 dB from the AGC: 34 dB without the   */
/* cap. The cap must hold the 50 ms output RMS at most 24.1 dB over the input RMS.                  */
/*                                                                                                  */
/* Two measures, both from the first hop of the tone:                                               */
/*   post one-pole   50 ms one-pole mean squares of the post output y and of x_d at 16 kHz, hop by  */
/*                   hop (the contract's own definition in step 5)                                  */
/*   engine 50 ms    48 kHz output vs input, aligned by es_engine_latency_ms, in 50 ms rectangular  */
/*                   windows that start every 1 ms                                                  */
/* Limits (DECISIONS L 2026-11-08, "Gain cap onset"): post step 5 stays the spec formula.           */
/*   steady  hops and windows that start 200 ms (four 50 ms time constants) or more into the tone:  */
/*           the contract limit, 24.1 dB                                                            */
/*   onset   hops and windows that start in the first 200 ms after the onset: 25.6 dB. The filter's */
/*           3 taps raise the gain by 9.7 dB over 2 hops, faster than the 50 ms averages follow,    */
/*           so the output runs about 1 dB over +24 dB until they catch up (measured 25.05 dB in    */
/*           the worst 50 ms window, 25.45 dB on the one-pole). The limiter and soft clip at -3     */
/*           dBFS remain the hearing-safety bound on the absolute level.                            */
/* ---------------------------------------------------------------------------------------------- */
#define GAIN_CAP_LIMIT_DB 24.1          /* contract 14.1 Post row: steady state */
#define GAIN_CAP_ONSET_LIMIT_DB 25.6    /* first 200 ms after an onset (DECISIONS L 2026-11-08) */

static void test_gain_cap(void) {
  es_model *m = NULL;
  es_model_synthetic(11, 0, &m);
  for (int k = 0; k < 16; k++) {                   /* fc2 weight 0, bias 20: 1.5 tanh(20) = 1.5 exactly */
    int len = 2 * m->band_width[k] * m->band_taps[k];
    float *w = (float *)m->dec[k].fc2.w, *b = (float *)m->dec[k].fc2.b;
    memset(w, 0, sizeof(float) * 128 * (size_t)len);
    for (int j = 0; j < len; j++) b[j] = 20.0f;
  }
  float *hw = (float *)m->heads.w, *hb = (float *)m->heads.b;   /* p_target ~ 1, p_self ~ 0 */
  memset(hw, 0, sizeof(float) * 64 * 2);
  hb[0] = 10.0f;
  hb[1] = -10.0f;

  es_engine *e = make_engine(m, 48000, 64, 43);
  es_params p;
  es_params_default(&p);                           /* gate, duck and AGC on, room off */
  p.voice_db = 6.0f;
  p.max_gain_db = 18.0f;
  es_engine_set_params(e, &p);
  fill_slot0(e);
  e->post.agc_db = 18.0f;                          /* AGC held at its maximum from the start */

  int rate = 48000, t0 = rate / 2, total = 3 * rate;
  float *in = calloc((size_t)total, sizeof(float)), *out = malloc(sizeof(float) * (size_t)total);
  float amp = 0.001f * sqrtf(2.0f);                /* -60 dBFS RMS: the AGC wants more than 18 dB */
  for (int i = t0; i < total; i++) in[i] = amp * sinf(2.0f * (float)ES_PI * 1000.0f * (float)(i - t0) / (float)rate);

  /* Post one-pole measure. With 64-frame callbacks at most one hop runs per callback, so e->y and
     e->xd are that hop. */
  float agc_min = 99.0f;
  double a50 = 1.0 - exp(-4.0 / 50.0), in_ms = 0, out_ms = 0, post_onset = -999, post_steady = -999;
  int tone_hops = 0;
  for (int off = 0; off < total; off += 64) {
    uint64_t h0 = atomic_load(&e->s_hops);
    drive(e, in + off, out + off, 64);
    agc_min = fminf(agc_min, e->post.agc_db);
    if (atomic_load(&e->s_hops) == h0) continue;
    double mx = 0, my = 0;
    for (int i = 0; i < ES_HOP; i++) {
      mx += (double)e->xd[i] * e->xd[i] / ES_HOP;
      my += (double)e->y[i] * e->y[i] / ES_HOP;
    }
    in_ms += a50 * (mx - in_ms);
    out_ms += a50 * (my - out_ms);
    if (mx > 0 || tone_hops > 0) {                 /* from the first hop whose x_d holds the tone */
      double g = 10.0 * log10(out_ms / in_ms);
      if (tone_hops >= 50) post_steady = fmax(post_steady, g);   /* 50 hops = 200 ms */
      else post_onset = fmax(post_onset, g);
      tone_hops++;
    }
  }

  /* Engine 50 ms measure. */
  int D = (int)lround(es_engine_latency_ms(e) * rate / 1000.0);   /* output lag in samples */
  int W = rate / 20;                               /* 50 ms */
  double eng_steady = -999, eng_onset = -999, last = 0;
  for (int s = t0; s + W + D <= total; s += 48) {
    double p_in = 0, p_out = 0;
    for (int i = 0; i < W; i++) {
      p_in += (double)in[s + i] * in[s + i];
      p_out += (double)out[s + D + i] * out[s + D + i];
    }
    double g = 10.0 * log10((p_out + 1e-30) / (p_in + 1e-30));
    if (s >= t0 + rate / 5) eng_steady = fmax(eng_steady, g);
    else eng_onset = fmax(eng_onset, g);
    last = g;
  }
  printf("INFO gain cap: post one-pole worst %.4f dB in the first 200 ms, %.4f dB after (%d hops); "
         "engine 50 ms windows worst %.4f dB in the first 200 ms, %.4f dB after, last %.4f dB; "
         "cap %.2f dB, AGC min %.2f dB\n",
         post_onset, post_steady, tone_hops, eng_onset, eng_steady, last, e->post.cap_db, agc_min);
  check_le("gain_cap_steady_post_50ms_onepole_output_over_xd_db", post_steady, GAIN_CAP_LIMIT_DB);
  check_le("gain_cap_steady_engine_50ms_rms_output_over_input_db", eng_steady, GAIN_CAP_LIMIT_DB);
  check_le("gain_cap_onset_post_50ms_onepole_output_over_xd_db", post_onset, GAIN_CAP_ONSET_LIMIT_DB);
  check_le("gain_cap_onset_engine_50ms_rms_output_over_input_db", eng_onset, GAIN_CAP_ONSET_LIMIT_DB);
  check_true("gain_cap_engaged_agc_at_max", last > 23.5 && e->post.cap_db < -9.0f && agc_min >= 17.99f);
  free(in);
  free(out);
  es_engine_free(e);
  es_model_free(m);
}

/* ---------------------------------------------------------------------------------------------- */
/* Hand-off race (contract 4.3): a writer thread calls set_cond 1,000,000 times while another      */
/* thread renders and runs hops. Every accepted change must be seen by the worker exactly once and  */
/* resets must happen exactly for the buffers that asked for one.                                   */
/* ---------------------------------------------------------------------------------------------- */
typedef struct {
  es_engine *e;
  _Atomic int done;
  uint64_t calls, busy, accepted, want_resets;
} handoff_job;

static void spin_us(double us) {
  static mach_timebase_info_data_t tb;
  if (tb.denom == 0) mach_timebase_info(&tb);
  uint64_t t0 = mach_absolute_time(), ticks = (uint64_t)(us * 1000.0 * tb.denom / tb.numer);
  while (mach_absolute_time() - t0 < ticks) {}
}

static void *handoff_writer(void *arg) {
  handoff_job *j = arg;
  es_cond *c = es_cond_new();
  c->n_tokens = 20;
  for (int i = 0; i < 4096; i++) c->film[i] = 0.01f * urand();
  int active_empty = 1;                            /* every slot of a new engine starts empty */
  uint32_t r = 12345u;
  for (j->calls = 0; j->calls < 1000000; j->calls++) {
    r ^= r << 13; r ^= r >> 17; r ^= r << 5;
    int use_null = (r & 3u) == 0, reset = (int)((r >> 2) & 1u);
    es_status st = es_engine_set_cond(j->e, 0, use_null ? NULL : c, reset);
    if (st == ES_ERR_BUSY) {
      j->busy++;
      spin_us(2.0);
    } else if (st == ES_OK) {
      j->accepted++;
      if (reset || active_empty) j->want_resets++;   /* the 20-engine 8.5 reset rule */
      active_empty = use_null;
    }
  }
  atomic_store(&j->done, 1);
  es_cond_free(c);
  return NULL;
}

static void *handoff_worker(void *arg) {
  handoff_job *j = arg;
  float in[192], out[192];
  while (!atomic_load(&j->done)) {
    for (int i = 0; i < 192; i++) in[i] = 0.1f * urand();
    drive(j->e, in, out, 192);                       /* 192 frames at 48 kHz = one hop */
  }
  for (int k = 0; k < 10; k++) drive(j->e, in, out, 192);   /* pick up the last change */
  return NULL;
}

static void test_handoff(const es_model *ap) {
  handoff_job j;
  memset(&j, 0, sizeof(j));
  j.e = make_engine(ap, 48000, 192, 43);
  pthread_t tw, tk;
  pthread_create(&tk, NULL, handoff_worker, &j);
  pthread_create(&tw, NULL, handoff_writer, &j);
  pthread_join(tw, NULL);
  pthread_join(tk, NULL);
  es_engine *e = j.e;
  printf("INFO handoff: %llu calls, %llu busy, %llu accepted, %llu switches seen, %llu resets (want %llu)\n",
         (unsigned long long)j.calls, (unsigned long long)j.busy, (unsigned long long)j.accepted,
         (unsigned long long)e->n_switches[0], (unsigned long long)e->n_resets[0],
         (unsigned long long)j.want_resets);
  check_true("handoff_busy_seen", j.busy > 0 && j.accepted > 1000);
  check_true("handoff_every_change_applied_once", e->n_switches[0] == j.accepted &&
             atomic_load(&e->active[0]) == atomic_load(&e->ack[0]));
  check_true("handoff_resets_only_for_their_buffer", e->n_resets[0] == j.want_resets);
  es_engine_free(e);
}

/* A reset asked for while no worker runs must survive a second set_cond before the worker restarts
   (pause, switch person with reset, update the voiceprint warm, resume). es_engine_set_manual_worker
   (e, 0) plays the stopped worker: set_cond then acknowledges its own changes. */
static void test_reset_kept_while_stopped(const es_model *ap) {
  es_engine *e = make_engine(ap, 48000, 192, 43);
  plain_params(e, 1);
  es_cond *c = es_cond_new();
  c->n_tokens = 20;
  float in[192] = { 0 }, out[192];
  es_engine_set_cond(e, 0, c, 0);                  /* wakes the slot: one reset */
  for (int k = 0; k < 5; k++) drive(e, in, out, 192);
  uint64_t r0 = e->n_resets[0];

  es_engine_set_manual_worker(e, 0);               /* the worker stops */
  es_status s1 = es_engine_set_cond(e, 0, c, 1);   /* new person: reset */
  es_status s2 = es_engine_set_cond(e, 0, c, 0);   /* then a warm update of the same person */
  es_engine_set_manual_worker(e, 1);               /* the worker resumes */
  for (int k = 0; k < 5; k++) drive(e, in, out, 192);
  uint64_t kept = e->n_resets[0] - r0;

  es_engine_set_manual_worker(e, 0);               /* control: two warm updates while stopped */
  es_status s3 = es_engine_set_cond(e, 0, c, 0);
  es_status s4 = es_engine_set_cond(e, 0, c, 0);
  es_engine_set_manual_worker(e, 1);
  for (int k = 0; k < 5; k++) drive(e, in, out, 192);
  uint64_t extra = e->n_resets[0] - r0 - kept;
  printf("INFO reset while stopped: %llu reset(s) applied after resume (want 1), %llu for two warm updates (want 0)\n",
         (unsigned long long)kept, (unsigned long long)extra);
  check_true("handoff_reset_kept_while_stopped", s1 == ES_OK && s2 == ES_OK && s3 == ES_OK && s4 == ES_OK &&
             kept == 1 && extra == 0);
  es_cond_free(c);
  es_engine_free(e);
}

/* 20-engine 8.3 step 3: at most one underrun count per callback. A 1,024-frame callback on an engine
   with max_frames 256 runs as 4 chunks; with no hops it starves in several of them, which must still
   count once (two counts would start a slack step on their own). */
static void test_underrun_once_per_callback(const es_model *ap) {
  es_engine_config cfg = { 48000, 64, 256, 1, 43 };
  es_engine *e = NULL;
  int ok = es_engine_new(ap, &cfg, &e) == ES_OK;
  if (!ok) {
    check_true("underrun_once_per_callback", 0);
    return;
  }
  es_engine_set_manual_worker(e, 1);
  es_engine_set_mute(e, 0);
  plain_params(e, 1);
  static float in[1024], out[1024];
  for (int i = 0; i < 1024; i++) in[i] = 0.1f * sinf(2.0f * (float)ES_PI * 500.0f * (float)i / 48000.0f);
  for (int k = 0; k < 200; k++) drive(e, in, out, 64);   /* steady state, no underruns */
  uint64_t u0 = underruns(e);
  es_engine_render(e, in, out, 1024);              /* no hops: starves */
  uint64_t du = underruns(e) - u0;
  int slack = atomic_load(&e->s_slack16);
  printf("INFO underrun per callback: %llu count(s) for one starved 4-chunk callback, slack16 %d (want 43), "
         "grow_pending %d\n", (unsigned long long)du, slack, e->grow_pending);
  check_true("underrun_once_per_callback", u0 == 0 && du == 1 && slack == 43 && e->grow_pending == 0);
  es_engine_free(e);
}

/* ---------------------------------------------------------------------------------------------- */
/* Slack step (contract 4.2 "Slack growth", 20-engine 8.3 steps 5 and 6)                           */
/* ---------------------------------------------------------------------------------------------- */
/* Two real underruns within 60 s (the worker is held back until the output runs dry, twice) raise
   slack16 from 43 to 64; latency grows by 21 / 16 kHz = 1.3125 ms and an impulse afterwards shows
   the new delay. */
static void test_slack_from_underruns(const es_model *ap) {
  int rate = 48000, frames = 64;
  es_engine *e = make_engine(ap, rate, frames, 43);
  plain_params(e, 1);
  float lat0 = es_engine_latency_ms(e);
  float in[64], out[64];
  long t = 0;
  for (int k = 0; k < 450; k++, t += 64) {         /* 0.6 s */
    for (int i = 0; i < 64; i++) in[i] = 0.25f * sinf(2.0f * (float)ES_PI * 500.0f * (float)(t + i) / (float)rate);
    drive(e, in, out, 64);
  }
  for (int round = 1; round <= 2; round++) {
    for (int k = 0; k < 200 && underruns(e) < (uint64_t)round; k++, t += 64) {   /* no hops: run dry */
      for (int i = 0; i < 64; i++) in[i] = 0.25f * sinf(2.0f * (float)ES_PI * 500.0f * (float)(t + i) / (float)rate);
      es_engine_render(e, in, out, 64);
    }
    for (int k = 0; k < 225; k++, t += 64) {       /* 0.3 s normal */
      for (int i = 0; i < 64; i++) in[i] = 0.25f * sinf(2.0f * (float)ES_PI * 500.0f * (float)(t + i) / (float)rate);
      drive(e, in, out, 64);
    }
  }
  es_stats s;
  es_engine_get_stats(e, &s);
  float lat1 = es_engine_latency_ms(e);
  check_true("slack_step_two_underruns_raise_43_to_64", s.underruns == 2 && s.slack16 == 64 && e->ins == 21);
  check_le("slack_step_latency_plus_1_3125_ms", fabs((lat1 - lat0) - 1.3125), 1e-4);

  /* Silence, then an impulse: the measured delay is the new latency. */
  int total = (int)(1.1 * rate), i0 = (int)(0.6 * rate);
  float *xi = calloc((size_t)total, sizeof(float)), *yo = malloc(sizeof(float) * (size_t)total);
  xi[i0] = 0.5f;
  run_buffer(e, xi, yo, total, frames);
  double est = group_delay_ms(yo + i0, rate);
  check("slack_step_impulse_delay_is_new_latency", fabs(est - lat1) <= 0.0625 && underruns(e) == 2,
        fabs(est - lat1), 0.0625);
  free(xi);
  free(yo);
  es_engine_free(e);
}

/* The 21 planned zeros themselves: a slack step decided in steady state (the render's grow_pending
   set by the test, as two underruns would) must ramp down over 1 ms, insert the zeros, and ramp up
   over 1 ms. On a 500 Hz tone of amplitude A, no sample-to-sample step may exceed the tone's own
   slope plus the ramp's (A (2 pi 500 / sr + 1 / ramp_len), with 25% margin); a hard gap would step
   by up to A. */
static void test_slack_ramps(const es_model *ap, int rate, int frames) {
  es_engine *e = make_engine(ap, rate, frames, 43);
  plain_params(e, 1);
  const float A = 0.25f;
  int total = rate;                                /* 1 s */
  float *in = malloc(sizeof(float) * (size_t)total), *out = malloc(sizeof(float) * (size_t)total);
  for (int i = 0; i < total; i++) in[i] = A * sinf(2.0f * (float)ES_PI * 500.0f * (float)i / (float)rate);
  int at = -1, start = (int)(0.6 * rate), forced = 0;
  for (int off = 0; off < total; off += frames) {
    int n = total - off < frames ? total - off : frames;
    if (off >= start && !forced) {
      e->grow_pending = ES_SLACK_STEP;             /* what two underruns within 60 s would set */
      forced = 1;
    }
    int64_t ins_before = e->ins;
    drive(e, in + off, out + off, n);
    if (at < 0 && e->ins != ins_before) at = off;  /* the callback that started with the zeros */
  }
  double slope = A * (2.0 * ES_PI * 500.0 / rate + 1.0 / e->ramp1ms) * 1.25, worst = 0, quiet = 1;
  int lo = at - rate / 50, hi = at + rate / 50;    /* +/- 20 ms around the inserted zeros */
  if (lo < 1) lo = 1;
  if (hi > total) hi = total;
  for (int i = lo; i < hi; i++) {
    worst = fmax(worst, fabs(out[i] - out[i - 1]));
    quiet = fmin(quiet, fabs(out[i]) + fabs(out[i - 1]) + fabs(out[i + 1 < total ? i + 1 : i]));
  }
  char name[96];
  snprintf(name, sizeof(name), "slack_step_ramps_no_hard_step_%d_%d", rate, frames);
  check(name, at > 0 && worst <= slope, worst, slope);
  snprintf(name, sizeof(name), "slack_step_zeros_inserted_%d_%d", rate, frames);
  check(name, at > 0 && quiet < 1e-4 && atomic_load(&e->s_slack16) == 64, quiet, 1e-4);
  free(in);
  free(out);
  es_engine_free(e);
}

int main(void) {
  printf("test_latency (package E-2)\n");
  es_model *ap = NULL;
  if (es_model_synthetic(3, 1, &ap) != ES_OK) {
    printf("FAIL synthetic_model 0 1\n");
    return 1;
  }
  test_impulse(ap);
  test_alignment(ap);
  test_gain_cap();
  test_handoff(ap);
  test_reset_kept_while_stopped(ap);
  test_underrun_once_per_callback(ap);
  test_slack_from_underruns(ap);
  test_slack_ramps(ap, 48000, 64);
  test_slack_ramps(ap, 48000, 1104);
  test_slack_ramps(ap, 44100, 480);
  es_model_free(ap);
  printf("test_latency: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
