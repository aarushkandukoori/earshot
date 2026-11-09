/*
 * earshot_stub.c - a pass-through stand-in for the real C engine (30-ios 2.5).
 *
 * Why it exists: the app has to link and run before WP-ENGINE finishes the real
 * engine (step E-2). This file implements EVERY function declared in
 * engine/include/earshot.h, so Swift and AudioHost.c can be written and tested
 * today. The lead deletes it at integration (`make project ENGINE=real`).
 *
 * What it really does:
 *   - Model: accepts any buffer; hashes are 64 zeros (that is how the app knows
 *     it is running the stub); the self-test always passes with error 0.
 *   - es_vad: the real contract 2.4 energy rule (20-engine 7.1), bit for bit.
 *   - es_enroll: counts tokens with the real "groups of three" rule and makes a
 *     unit vector g from the clip's band energies (not a real voiceprint).
 *   - Engine: starts muted, plays the mic straight through with the real mute
 *     ramp (5 ms down, 400 ms raised-cosine fade-in) and a -3 dBFS limiter, keeps
 *     a 16 kHz history of the mic (for enrollment taps), plays cue beeps, and
 *     runs the latency probe (sweeps out, mic in, GCC-PHAT) so Phase 0 can measure
 *     the route before E-2.
 *   - There is NO worker thread: start/stop worker only flip a flag.
 *
 * Real-time rule: es_engine_render runs on the audio thread, so it never
 * allocates, locks, logs or does file I/O. Everything it needs is allocated in
 * es_engine_new.
 */
#include "earshot.h"

#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define STUB_PI 3.14159265358979323846

/* ------------------------------------------------------------------------- */
/* Status strings                                                             */
/* ------------------------------------------------------------------------- */

const char *es_status_str(es_status s) {
  switch (s) {
    case ES_OK: return "ok";
    case ES_ERR_ARG: return "bad argument";
    case ES_ERR_FORMAT: return "bad file format";
    case ES_ERR_VERSION: return "wrong version";
    case ES_ERR_CONFIG: return "config mismatch";
    case ES_ERR_NOMEM: return "out of memory";
    case ES_ERR_ENROLL_SHORT: return "not enough speech";
    case ES_ERR_BUSY: return "busy";
    case ES_ERR_STATE: return "wrong state";
    case ES_ERR_RATE: return "unsupported sample rate";
    case ES_ERR_PROBE_WEAK: return "probe too weak";
    case ES_ERR_SELFTEST: return "self-test failed";
    case ES_ERR_THREAD: return "thread setup failed";
  }
  return "unknown status";
}

/* ------------------------------------------------------------------------- */
/* 1. Model (the stub keeps no weights at all)                                */
/* ------------------------------------------------------------------------- */

struct es_model {
  char config_hash[65]; /* 64 '0' characters plus the terminating NUL */
  char data_hash[65];
};

es_status es_model_load(const void *esw, size_t nbytes, es_model **out, char *err, size_t errlen) {
  (void)esw;
  (void)nbytes; /* the stub accepts any buffer, even an empty one */
  if (out == NULL) return ES_ERR_ARG;
  es_model *m = (es_model *)calloc(1, sizeof(es_model));
  if (m == NULL) return ES_ERR_NOMEM;
  for (int i = 0; i < 64; i++) {
    m->config_hash[i] = '0';
    m->data_hash[i] = '0';
  }
  if (err != NULL && errlen > 0) err[0] = '\0'; /* no error text */
  *out = m;
  return ES_OK;
}

void es_model_free(es_model *m) { free(m); }

const char *es_model_config_hash(const es_model *m) { return m ? m->config_hash : ""; }

const char *es_model_data_hash(const es_model *m) { return m ? m->data_hash : ""; }

int es_model_latency_samples(const es_model *m) {
  (void)m;
  return ES_ALGO_LATENCY;
}

es_status es_selftest(const es_model *m, const void *est, size_t nbytes, float *max_err_out) {
  (void)est;
  (void)nbytes;
  if (m == NULL) return ES_ERR_ARG;
  if (max_err_out != NULL) *max_err_out = 0.0f; /* the stub always "passes" */
  return ES_OK;
}

/* ------------------------------------------------------------------------- */
/* 2. Enrollment: the real VAD, a fake voiceprint                             */
/* ------------------------------------------------------------------------- */

struct es_voiceprint {
  int n_tok;            /* tokens after the 48 / 16 limit */
  float g[ES_G_DIM];    /* unit vector (fake: from band energies) */
  float e[ES_WESPK_DIM];/* copy of the WeSpeaker vector, if given */
  int has_wespk;
  int role;
  float speech_s;       /* kept frames x 0.032 */
};

es_voiceprint *es_voiceprint_new(void) { return (es_voiceprint *)calloc(1, sizeof(es_voiceprint)); }

void es_voiceprint_free(es_voiceprint *v) { free(v); }

int es_enroll_num_frames(int n_samples) {
  if (n_samples < 576) return 0; /* deliberate minimum, so F starts at 2 */
  return (n_samples - 64) / 512 + 1;
}

/* qsort helper: ascending doubles */
static int stub_cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

/* Contract 2.4 energy VAD, written exactly like 20-engine 7.1 (double math). */
void es_vad(const float *pcm, int n, uint8_t *mask_out) {
  int F = es_enroll_num_frames(n);
  if (F == 0 || pcm == NULL || mask_out == NULL) return; /* F = 0 writes nothing */

  double *E = (double *)malloc(sizeof(double) * (size_t)F);
  double *sorted = (double *)malloc(sizeof(double) * (size_t)F);
  uint8_t *raw = (uint8_t *)malloc((size_t)F);
  if (E == NULL || sorted == NULL || raw == NULL) {
    memset(mask_out, 0, (size_t)F); /* out of memory: say "no speech" */
    free(E); free(sorted); free(raw);
    return;
  }

  /* Energy of each 512-sample frame of the clip padded with 448 zeros in front. */
  for (int f = 0; f < F; f++) {
    double sum = 0.0;
    for (int i = 0; i < 512; i++) {
      long j = 512L * f + i - 448;
      double x = j >= 0 ? (double)pcm[j] : 0.0;
      sum += x * x;
    }
    E[f] = 10.0 * log10(sum / 512.0 + 1e-10);
  }

  /* Threshold = max(10th percentile + 6 dB, loudest - 40 dB). */
  memcpy(sorted, E, sizeof(double) * (size_t)F);
  qsort(sorted, (size_t)F, sizeof(double), stub_cmp_double);
  double p10 = sorted[(F - 1) / 10];
  double emax = sorted[F - 1];
  double thr = fmax(p10 + 6.0, emax - 40.0);

  for (int f = 0; f < F; f++) raw[f] = E[f] > thr ? 1 : 0;

  /* Dilate by one frame on each side (out-of-range neighbours count as 0). */
  for (int f = 0; f < F; f++) {
    uint8_t left = f > 0 ? raw[f - 1] : 0;
    uint8_t right = f + 1 < F ? raw[f + 1] : 0;
    mask_out[f] = (uint8_t)(left | raw[f] | right);
  }
  free(E);
  free(sorted);
  free(raw);
}

/* Fake g: log energy of the clip in 128 bands (Goertzel at 62.5 Hz steps,
 * at most the first 10 s), mean removed, then scaled to unit length.
 * It only has to be a stable unit vector that differs between clips. */
static void stub_band_g(const float *pcm, int n, float g[ES_G_DIM]) {
  int len = n < 160000 ? n : 160000;
  double v[ES_G_DIM];
  double mean = 0.0;
  for (int b = 0; b < ES_G_DIM; b++) {
    double freq = 62.5 * (b + 1);                       /* 62.5 Hz .. 8 kHz */
    double coeff = 2.0 * cos(2.0 * STUB_PI * freq / ES_SR);
    double s1 = 0.0, s2 = 0.0;
    for (int i = 0; i < len; i++) {                      /* Goertzel recursion */
      double s0 = (double)pcm[i] + coeff * s1 - s2;
      s2 = s1;
      s1 = s0;
    }
    double power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    v[b] = log(power + 1e-10);
    mean += v[b];
  }
  mean /= ES_G_DIM;
  double norm = 0.0;
  for (int b = 0; b < ES_G_DIM; b++) {
    v[b] -= mean;
    norm += v[b] * v[b];
  }
  norm = sqrt(norm);
  for (int b = 0; b < ES_G_DIM; b++) g[b] = norm > 1e-12 ? (float)(v[b] / norm) : 0.0f;
  if (norm <= 1e-12) g[0] = 1.0f; /* silence: still return a unit vector */
}

/* Counts kept frames and tokens with the real grouping rule (20-engine 7.2 step 4). */
static int stub_count_tokens(const uint8_t *mask, int F, int *kept_out) {
  int kept = 0;
  for (int f = 0; f < F; f++) kept += mask[f] ? 1 : 0;
  if (kept_out != NULL) *kept_out = kept;
  return kept / 3; /* groups of three from the start; a final 1 or 2 is dropped */
}

/* Shared checks of es_enroll and es_embed_g: pointers, length, finite samples. */
static es_status stub_check_pcm(const float *pcm, int n) {
  if (pcm == NULL || n < 0 || n > 960000) return ES_ERR_ARG;
  for (int i = 0; i < n; i++)
    if (!isfinite(pcm[i])) return ES_ERR_ARG;
  return ES_OK;
}

es_status es_enroll(const es_model *m, const float *pcm, int n, const uint8_t *mask, es_role role,
                    const float *e_wespk, es_voiceprint *out) {
  if (m == NULL || out == NULL) return ES_ERR_ARG;
  if (role != ES_ROLE_TARGET && role != ES_ROLE_OWN) return ES_ERR_ARG;
  es_status st = stub_check_pcm(pcm, n);
  if (st != ES_OK) return st;
  if (e_wespk != NULL)
    for (int i = 0; i < ES_WESPK_DIM; i++)
      if (!isfinite(e_wespk[i])) return ES_ERR_ARG;

  int F = es_enroll_num_frames(n);
  if (F == 0) return ES_ERR_ENROLL_SHORT;

  /* Use the caller's mask, or run the VAD when it is NULL. */
  uint8_t *own_mask = NULL;
  const uint8_t *use = mask;
  if (use == NULL) {
    own_mask = (uint8_t *)malloc((size_t)F);
    if (own_mask == NULL) return ES_ERR_NOMEM;
    es_vad(pcm, n, own_mask);
    use = own_mask;
  }
  int kept = 0;
  int n_avail = stub_count_tokens(use, F, &kept);
  free(own_mask);
  if (n_avail < 4) return ES_ERR_ENROLL_SHORT;

  int limit = role == ES_ROLE_TARGET ? ES_MAX_TGT_TOKENS : ES_MAX_OWN_TOKENS;
  out->n_tok = n_avail < limit ? n_avail : limit;
  out->role = (int)role;
  out->speech_s = (float)kept * 0.032f;
  stub_band_g(pcm, n, out->g);
  if (e_wespk != NULL) {
    memcpy(out->e, e_wespk, sizeof(out->e));
    out->has_wespk = 1;
  } else {
    memset(out->e, 0, sizeof(out->e));
    out->has_wespk = 0;
  }
  return ES_OK;
}

int es_voiceprint_num_tokens(const es_voiceprint *v) { return v ? v->n_tok : 0; }

void es_voiceprint_get_g(const es_voiceprint *v, float g_out[ES_G_DIM]) {
  if (v == NULL || g_out == NULL) return;
  memcpy(g_out, v->g, sizeof(float) * ES_G_DIM);
}

int es_voiceprint_has_wespk(const es_voiceprint *v) { return v ? v->has_wespk : 0; }

float es_voiceprint_speech_seconds(const es_voiceprint *v) { return v ? v->speech_s : 0.0f; }

es_status es_embed_g(const es_model *m, const float *pcm, int n, float g_out[ES_G_DIM]) {
  if (m == NULL || g_out == NULL) return ES_ERR_ARG;
  es_status st = stub_check_pcm(pcm, n);
  if (st != ES_OK) return st;
  int F = es_enroll_num_frames(n);
  if (F == 0) return ES_ERR_ENROLL_SHORT;
  uint8_t *mask = (uint8_t *)malloc((size_t)F);
  if (mask == NULL) return ES_ERR_NOMEM;
  es_vad(pcm, n, mask);
  int n_avail = stub_count_tokens(mask, F, NULL);
  free(mask);
  if (n_avail < 4) return ES_ERR_ENROLL_SHORT;
  stub_band_g(pcm, n, g_out);
  return ES_OK;
}

/* Conditioning: the stub "copies nothing"; it only remembers mode and token count. */
struct es_cond {
  int n_tokens; /* target + own + 4 null tokens; 0 = empty */
  int mode;
};

es_cond *es_cond_new(void) { return (es_cond *)calloc(1, sizeof(es_cond)); }

void es_cond_free(es_cond *c) { free(c); }

es_status es_cond_build(const es_model *m, es_mode mode, const es_voiceprint *target, const es_voiceprint *own,
                        int use_wespk, es_cond *out) {
  (void)use_wespk;
  if (m == NULL || out == NULL) return ES_ERR_ARG;
  if (mode != ES_MODE_PERSON && mode != ES_MODE_EVERYONE) return ES_ERR_ARG;
  /* target must be NULL exactly when the mode is EVERYONE */
  if ((mode == ES_MODE_EVERYONE) != (target == NULL)) return ES_ERR_ARG;
  int n_t = target ? target->n_tok : 0;
  int n_o = own ? own->n_tok : 0;
  out->n_tokens = n_t + n_o + ES_NULL_TOKENS;
  out->mode = (int)mode;
  return ES_OK;
}

/* ------------------------------------------------------------------------- */
/* 3. Streams: copy in to out (no model)                                      */
/* ------------------------------------------------------------------------- */

struct es_stream {
  int unused;
};

es_stream *es_stream_new(const es_model *m) {
  if (m == NULL) return NULL;
  return (es_stream *)calloc(1, sizeof(es_stream));
}

void es_stream_free(es_stream *s) { free(s); }

void es_stream_reset(es_stream *s) { (void)s; }

int es_stream_hop(es_stream *s, const es_cond *c, const float in[ES_HOP], float out[ES_HOP], float heads[2]) {
  (void)s;
  (void)c;
  memcpy(out, in, sizeof(float) * ES_HOP);
  heads[0] = 0.0f;
  heads[1] = 0.0f;
  return 0;
}

/* ------------------------------------------------------------------------- */
/* 5. DSP blocks (exactly 20-engine 11; the engine section below uses them)   */
/* ------------------------------------------------------------------------- */

void es_ramp_init(es_ramp *r, float sr, float gain) {
  r->gain = gain;
  r->start = gain;
  r->target = gain;
  r->step = 0.0f;
  r->phase = 1.0f;
  r->sr = sr;
  r->shape = 0;
}

void es_ramp_to(es_ramp *r, float target, float ms, int shape) {
  r->shape = shape;
  if (ms <= 0.0f) { /* jump */
    r->gain = target;
    r->start = target;
    r->target = target;
    r->phase = 1.0f;
    r->step = 0.0f;
    return;
  }
  r->start = r->gain; /* continue from where we are, never jump */
  r->target = target;
  r->phase = 0.0f;
  r->step = 1000.0f / (ms * r->sr);
}

void es_ramp_process(es_ramp *r, float *buf, int n) {
  for (int i = 0; i < n; i++) {
    if (r->phase < 1.0f) {
      r->phase += r->step;
      if (r->phase > 1.0f) r->phase = 1.0f;
      float w = r->shape == 1 ? 0.5f - 0.5f * cosf((float)STUB_PI * r->phase) : r->phase;
      r->gain = r->start + (r->target - r->start) * w;
    }
    buf[i] *= r->gain;
  }
}

void es_limiter_init(es_limiter *l, float sr, float ceiling_dbfs, float release_ms) {
  l->ceiling = powf(10.0f, ceiling_dbfs / 20.0f); /* -3 dBFS = 0.7079458 */
  l->rel_coef = expf(-1.0f / (release_ms * 0.001f * sr));
  l->gain = 1.0f;
  l->min_gain = 1e-6f;
}

void es_limiter_process(es_limiter *l, float *x, int n) {
  for (int i = 0; i < n; i++) {
    float a = fabsf(x[i]);
    float gt = a > l->ceiling ? l->ceiling / a : 1.0f;
    if (gt < l->gain) l->gain = gt;                     /* instant attack */
    else l->gain = gt + (l->gain - gt) * l->rel_coef;   /* release toward gt */
    if (l->gain < l->min_gain) l->gain = l->min_gain;
    x[i] *= l->gain;
  }
}

void es_softclip(float *buf, int n) {
  const float k = 0.8912509f; /* -1 dBFS */
  const float r = 1.0f - k;
  for (int i = 0; i < n; i++) {
    float a = fabsf(buf[i]);
    if (a > k) buf[i] = copysignf(k + r * tanhf((a - k) / r), buf[i]);
  }
}

void es_params_default(es_params *p) {
  p->voice_db = 0.0f;
  p->room_db = -60.0f;
  p->max_gain_db = 18.0f;
  p->out_gain_db = 0.0f;
  p->slot_gain[0] = 1.0f;
  p->slot_gain[1] = 0.0f;
  p->slot_gain[2] = 0.0f;
  p->xfade_ms = 150.0f;
  p->duck_own = 1;
  p->gate = 1;
  p->agc = 1;
  p->bypass = 0;
}

/* ------------------------------------------------------------------------- */
/* 4. Engine: pass-through with mute ramp, limiter, history, cues and probe   */
/* ------------------------------------------------------------------------- */

#define STUB_HIST_GUARD 16384      /* overrun guard of the history ring (20-engine 8.1) */
#define STUB_OUTHIST_MAX 48000     /* es_engine_read_output returns at most this many */
#define STUB_PROBE_SWEEPS 5
#define STUB_PROBE_NFFT 65536      /* GCC-PHAT FFT size (log2n = 16) */

enum { PROBE_IDLE = 0, PROBE_RUNNING = 1, PROBE_CAPTURED = 2 };

struct es_engine {
  es_engine_config cfg;
  es_params params;                      /* stored only; the stub's render ignores params */
  int slot_tokens[ES_MAX_INSTANCES];     /* conditioning token count per slot, 0 = empty */
  int worker_running;                    /* no real thread in the stub */

  /* 16 kHz mic history: written by render, read by app threads with a cursor */
  float *hist;                           /* ES_HISTORY_SAMPLES floats */
  _Atomic uint64_t hist_w;               /* total samples ever written */

  /* decimator to 16 kHz */
  float dec_sum;                         /* 48 kHz: running sum of up to 3 samples */
  int dec_count;
  uint64_t in_count;                     /* 44.1 kHz: input samples seen so far */
  uint64_t out_count;                    /* 44.1 kHz: 16 kHz samples produced so far */

  /* render scratch, allocated once */
  float *xin;                            /* sanitized input, max_frames */

  /* safety chain */
  es_ramp mute_ramp;
  int mute_applied;                      /* 1 = the render last applied "muted" */
  _Atomic int mute_req;                  /* written by es_engine_set_mute */
  es_limiter limiter;

  /* cue beeps */
  _Atomic int cue_req;                   /* 0 = no new request, else 1..3 beeps */
  int cue_beeps;                         /* beeps in the running sequence, 0 = none */
  int cue_pos;                           /* sample position inside the sequence */
  float cue_phase;

  /* meters */
  _Atomic int32_t last_frames;
  _Atomic float in_rms_db;
  _Atomic float out_rms_db;
  _Atomic uint64_t probe_runs;

  /* latency probe */
  float *sweep;                          /* one 0.3 s sweep at the I/O rate */
  int sweep_len, sweep_period, play_len; /* samples: 0.3 s, 0.8 s, 4.0 s */
  float *cap;                            /* 4.5 s mic capture */
  int cap_len;
  int probe_pos;                         /* written by render while RUNNING */
  _Atomic int probe_state;
  int result_ready;                      /* cached result, non-RT side only */
  es_status result_status;
  es_probe_result result;
};

/* Builds the probe sweep (20-engine 12): 300 Hz to 6 kHz exponential, 0.3 s,
 * amplitude ES_PROBE_DBFS, 5 ms raised-cosine edges. */
static void stub_make_sweep(float *x, int len, double sr) {
  const double f1 = 300.0, f2 = 6000.0, T = 0.3;
  const double A = pow(10.0, ES_PROBE_DBFS / 20.0);
  const double L = log(f2 / f1);
  int edge = (int)(0.005 * sr);
  for (int i = 0; i < len; i++) {
    double t = i / sr;
    double v = A * sin(2.0 * STUB_PI * f1 * T / L * (exp(t * L / T) - 1.0));
    double w = 1.0;
    if (i < edge) w = 0.5 - 0.5 * cos(STUB_PI * i / edge);
    if (len - 1 - i < edge) w = 0.5 - 0.5 * cos(STUB_PI * (len - 1 - i) / edge);
    x[i] = (float)(v * w);
  }
}

es_status es_engine_new(const es_model *m, const es_engine_config *cfg, es_engine **out) {
  if (m == NULL || cfg == NULL || out == NULL) return ES_ERR_ARG;
  if (cfg->io_rate != 48000 && cfg->io_rate != 44100) return ES_ERR_RATE;
  if (cfg->io_frames < 16 || cfg->io_frames > cfg->max_frames || cfg->max_frames > 4096) return ES_ERR_ARG;
  if (cfg->n_instances < 1 || cfg->n_instances > ES_MAX_INSTANCES) return ES_ERR_ARG;
  if (cfg->slack16 != 22 && cfg->slack16 != 43 && cfg->slack16 != 64) return ES_ERR_ARG;

  es_engine *e = (es_engine *)calloc(1, sizeof(es_engine));
  if (e == NULL) return ES_ERR_NOMEM;
  e->cfg = *cfg;
  es_params_default(&e->params);
  double sr = (double)cfg->io_rate;
  e->sweep_len = (int)lround(0.3 * sr);
  e->sweep_period = (int)lround(0.8 * sr);
  e->play_len = (int)lround(4.0 * sr);
  e->cap_len = (int)lround(4.5 * sr);

  e->hist = (float *)calloc(ES_HISTORY_SAMPLES, sizeof(float));
  e->xin = (float *)calloc((size_t)cfg->max_frames, sizeof(float));
  e->sweep = (float *)calloc((size_t)e->sweep_len, sizeof(float));
  e->cap = (float *)calloc((size_t)e->cap_len, sizeof(float));
  if (e->hist == NULL || e->xin == NULL || e->sweep == NULL || e->cap == NULL) {
    es_engine_free(e);
    return ES_ERR_NOMEM;
  }
  stub_make_sweep(e->sweep, e->sweep_len, sr);

  atomic_init(&e->hist_w, 0);
  /* Starts muted (contract 4): gain 0, nothing armed; the first unmute fades in. */
  es_ramp_init(&e->mute_ramp, (float)sr, 0.0f);
  e->mute_applied = 1;
  atomic_init(&e->mute_req, 1);
  es_limiter_init(&e->limiter, (float)sr, -3.0f, 50.0f);
  atomic_init(&e->cue_req, 0);
  atomic_init(&e->last_frames, 0);
  atomic_init(&e->in_rms_db, -100.0f);
  atomic_init(&e->out_rms_db, -100.0f);
  atomic_init(&e->probe_runs, 0);
  atomic_init(&e->probe_state, PROBE_IDLE);
  *out = e;
  return ES_OK;
}

void es_engine_free(es_engine *e) {
  if (e == NULL) return;
  free(e->hist);
  free(e->xin);
  free(e->sweep);
  free(e->cap);
  free(e);
}

es_status es_engine_start_worker(es_engine *e, void *os_workgroup) {
  (void)os_workgroup; /* the stub has no worker thread to join */
  if (e == NULL) return ES_ERR_ARG;
  e->worker_running = 1;
  return ES_OK;
}

void es_engine_stop_worker(es_engine *e) {
  if (e != NULL) e->worker_running = 0;
}

int es_engine_run_hops(es_engine *e) {
  (void)e;
  return 0; /* nothing to compute: render already plays the input */
}

es_status es_engine_set_cond(es_engine *e, int slot, const es_cond *c, int reset_state) {
  (void)reset_state;
  if (e == NULL || slot < 0 || slot >= e->cfg.n_instances) return ES_ERR_ARG;
  if (c != NULL && c->n_tokens < 4) return ES_ERR_ARG;
  e->slot_tokens[slot] = c ? c->n_tokens : 0; /* NULL empties the slot; never BUSY */
  return ES_OK;
}

void es_engine_set_params(es_engine *e, const es_params *p) {
  if (e != NULL && p != NULL) e->params = *p; /* stored only (main thread) */
}

void es_engine_set_mute(es_engine *e, int mute) {
  if (e != NULL) atomic_store_explicit(&e->mute_req, mute ? 1 : 0, memory_order_release);
}

void es_engine_cue(es_engine *e, int beeps) {
  if (e == NULL) return;
  if (beeps < 1) beeps = 1;
  if (beeps > 3) beeps = 3;
  atomic_store_explicit(&e->cue_req, beeps, memory_order_release);
}

void es_engine_get_stats(const es_engine *e, es_stats *s) {
  if (s == NULL) return;
  memset(s, 0, sizeof(*s));
  if (e == NULL) return;
  es_engine *me = (es_engine *)e; /* atomics need a non-const pointer */
  s->hops = atomic_load_explicit(&me->hist_w, memory_order_acquire) / ES_HOP;
  s->probe_runs = atomic_load_explicit(&me->probe_runs, memory_order_relaxed);
  s->slack16 = e->cfg.slack16;
  s->last_frames = atomic_load_explicit(&me->last_frames, memory_order_relaxed);
  s->in_rms_db = atomic_load_explicit(&me->in_rms_db, memory_order_relaxed);
  s->out_rms_db = atomic_load_explicit(&me->out_rms_db, memory_order_relaxed);
  /* p_target, p_self, compute times and the counters stay 0 in the stub */
}

float es_engine_latency_ms(const es_engine *e) {
  (void)e;
  return 0.0f; /* the stub adds no delay over a plain pass-through */
}

uint64_t es_engine_input_position(const es_engine *e) {
  if (e == NULL) return 0;
  return atomic_load_explicit(&((es_engine *)e)->hist_w, memory_order_acquire);
}

/* History read with a caller-owned cursor: 20-engine 8.1 as amended by DECISIONS L
 * (2026-11-07, E-0 review):
 *   - a cursor older than the oldest held sample starts at the oldest held sample;
 *   - a cursor AHEAD of the writer returns 0 and leaves *cursor unchanged (it must never
 *     be treated as "lapped", which used to jump it 15.36 s back);
 *   - the overrun check is `w_after - start > size - 16384` (the guard keeps the newest
 *     16,384 slots out of reach, because the writer may be filling them right now). */
int es_engine_read_input(es_engine *e, uint64_t *cursor, float *dst, int max) {
  if (e == NULL || cursor == NULL || dst == NULL || max <= 0) return 0;
  const uint64_t size = ES_HISTORY_SAMPLES;
  const uint64_t keep = size - STUB_HIST_GUARD;             /* readable span, 245,760 samples */
  uint64_t w = atomic_load_explicit(&e->hist_w, memory_order_acquire);
  if (*cursor > w) return 0;                                /* from the future: read nothing, keep the cursor */
  uint64_t oldest = w > keep ? w - keep : 0;
  uint64_t start = *cursor > oldest ? *cursor : oldest;     /* too old: start at the oldest held */
  uint64_t avail = w - start;                               /* start <= w here */
  uint64_t count = avail < (uint64_t)max ? avail : (uint64_t)max;
  for (uint64_t i = 0; i < count; i++) dst[i] = e->hist[(start + i) & (size - 1)];
  uint64_t w_after = atomic_load_explicit(&e->hist_w, memory_order_acquire);
  if (w_after - start > keep) {                             /* the writer may have overwritten the copy */
    *cursor = w_after - keep;
    return 0;
  }
  *cursor = start + count;
  return (int)count;
}

/* The stub's output is the input, so a slot's recent output is the newest input
 * history (active slot) or silence (empty slot). */
int es_engine_read_output(es_engine *e, int slot, float *dst, int n) {
  if (e == NULL || dst == NULL || n <= 0 || slot < 0 || slot >= e->cfg.n_instances) return 0;
  if (n > STUB_OUTHIST_MAX) n = STUB_OUTHIST_MAX;
  if (e->slot_tokens[slot] < 4) {
    memset(dst, 0, sizeof(float) * (size_t)n);
    return n;
  }
  uint64_t w = atomic_load_explicit(&e->hist_w, memory_order_acquire);
  uint64_t cursor = w > (uint64_t)n ? w - (uint64_t)n : 0;
  return es_engine_read_input(e, &cursor, dst, n);
}

int es_engine_read_timings(es_engine *e, uint16_t *us, int max) {
  (void)e;
  (void)us;
  (void)max;
  return 0; /* no worker, no per-hop timings */
}

/* --- render helpers (all RT-safe: plain loops over preallocated memory) --- */

/* Writes one 16 kHz sample into the history ring (local write index w). */
static inline void stub_hist_push(es_engine *e, uint64_t *w, float v) {
  e->hist[*w & (ES_HISTORY_SAMPLES - 1)] = v;
  *w += 1;
}

/* Decimates n sanitized samples to 16 kHz: plain 3:1 averaging at 48 kHz,
 * nearest-sample pick at 44.1 kHz (16 kHz sample k = input sample
 * round(k * 441 / 160)). No anti-alias filter: the stub only needs a usable
 * history for enrollment and alarm work. */
static void stub_decimate(es_engine *e, const float *x, int n) {
  uint64_t w = atomic_load_explicit(&e->hist_w, memory_order_relaxed); /* only render writes it */
  if (e->cfg.io_rate == 48000) {
    for (int i = 0; i < n; i++) {
      e->dec_sum += x[i];
      e->dec_count += 1;
      if (e->dec_count == 3) {
        stub_hist_push(e, &w, e->dec_sum / 3.0f);
        e->dec_sum = 0.0f;
        e->dec_count = 0;
      }
    }
  } else {
    for (int i = 0; i < n; i++) {
      uint64_t pick = (e->out_count * 441 + 80) / 160; /* input index of the next 16 kHz sample */
      if (pick == e->in_count) {
        stub_hist_push(e, &w, x[i]);
        e->out_count += 1;
      }
      e->in_count += 1;
    }
  }
  atomic_store_explicit(&e->hist_w, w, memory_order_release); /* publish to readers */
}

/* Adds the cue beeps: 60 ms of 880 Hz at -18 dBFS, 5 ms raised-cosine edges,
 * 90 ms of silence between beeps (20-engine 11). */
static void stub_add_cues(es_engine *e, float *y, int n) {
  int req = atomic_exchange_explicit(&e->cue_req, 0, memory_order_acq_rel);
  if (req > 0) { /* a new request restarts the sequence */
    e->cue_beeps = req;
    e->cue_pos = 0;
    e->cue_phase = 0.0f;
  }
  if (e->cue_beeps == 0) return;
  const float sr = (float)e->cfg.io_rate;
  const int beep_len = (int)(0.060f * sr);
  const int cycle_len = (int)(0.150f * sr); /* 60 ms beep + 90 ms gap */
  const int edge = (int)(0.005f * sr);
  const int total = e->cue_beeps * cycle_len - (cycle_len - beep_len);
  const float dphi = (float)(2.0 * STUB_PI * 880.0) / sr;
  for (int i = 0; i < n && e->cue_pos < total; i++) {
    int offset = e->cue_pos % cycle_len;
    if (offset < beep_len) {
      float env = 1.0f;
      if (offset < edge) env = 0.5f - 0.5f * cosf((float)STUB_PI * offset / edge);
      if (beep_len - 1 - offset < edge) env = 0.5f - 0.5f * cosf((float)STUB_PI * (beep_len - 1 - offset) / edge);
      y[i] += 0.1258925f * env * sinf(e->cue_phase);
    }
    e->cue_phase += dphi;
    if (e->cue_phase >= (float)(2.0 * STUB_PI)) e->cue_phase -= (float)(2.0 * STUB_PI);
    e->cue_pos += 1;
  }
  if (e->cue_pos >= total) e->cue_beeps = 0; /* sequence finished */
}

/* RMS of n samples in dBFS (floor -100). */
static float stub_rms_db(const float *x, int n) {
  if (n <= 0) return -100.0f;
  double sum = 0.0;
  for (int i = 0; i < n; i++) sum += (double)x[i] * x[i];
  double db = 10.0 * log10(sum / n + 1e-10);
  return db < -100.0 ? -100.0f : (float)db;
}

/* One chunk of at most max_frames samples. */
static void stub_render_chunk(es_engine *e, const float *in, float *out, int n) {
  /* 1. sanitize the mic input */
  for (int i = 0; i < n; i++) {
    float v = in[i];
    if (!isfinite(v)) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    if (v < -1.0f) v = -1.0f;
    e->xin[i] = v;
  }
  /* 2. 16 kHz history for taps and quick add */
  stub_decimate(e, e->xin, n);

  /* 3. pass-through */
  memcpy(out, e->xin, sizeof(float) * (size_t)n);

  /* 4. mute ramp: 5 ms linear down, 400 ms raised-cosine fade-in on every unmute */
  int want = atomic_load_explicit(&e->mute_req, memory_order_acquire);
  if (want != e->mute_applied) {
    if (want) es_ramp_to(&e->mute_ramp, 0.0f, 5.0f, 0);
    else es_ramp_to(&e->mute_ramp, 1.0f, 400.0f, 1);
    e->mute_applied = want;
  }
  es_ramp_process(&e->mute_ramp, out, n);

  /* 5. cues after the mute, so they are heard while muted */
  stub_add_cues(e, out, n);

  /* 6. -3 dBFS limiter, soft clip, hard clip */
  es_limiter_process(&e->limiter, out, n);
  es_softclip(out, n);
  for (int i = 0; i < n; i++) out[i] = fminf(fmaxf(out[i], -1.0f), 1.0f);

  /* 7. latency probe replaces the output and records the mic */
  if (atomic_load_explicit(&e->probe_state, memory_order_acquire) == PROBE_RUNNING) {
    float g = e->mute_ramp.gain;
    for (int i = 0; i < n; i++) {
      int pos = e->probe_pos;
      float s = 0.0f;
      if (pos < e->play_len) {
        int k = pos / e->sweep_period;      /* which sweep */
        int off = pos - k * e->sweep_period;
        if (k < STUB_PROBE_SWEEPS && off < e->sweep_len) s = e->sweep[off];
      }
      out[i] = s * g;
      if (pos < e->cap_len) e->cap[pos] = e->xin[i];
      e->probe_pos = pos + 1;
      if (e->probe_pos >= e->cap_len) {
        atomic_fetch_add_explicit(&e->probe_runs, 1, memory_order_relaxed);
        atomic_store_explicit(&e->probe_state, PROBE_CAPTURED, memory_order_release);
        for (int j = i + 1; j < n; j++) out[j] = 0.0f; /* rest of this chunk: silence */
        break;
      }
    }
  }
}

void es_engine_render(es_engine *e, const float *in, float *out, int n) {
  if (e == NULL || in == NULL || out == NULL || n <= 0) return;
  int done = 0;
  float in_db = -100.0f, out_db = -100.0f;
  while (done < n) { /* callbacks larger than max_frames run in chunks */
    int chunk = n - done;
    if (chunk > e->cfg.max_frames) chunk = e->cfg.max_frames;
    stub_render_chunk(e, in + done, out + done, chunk);
    in_db = stub_rms_db(e->xin, chunk);
    out_db = stub_rms_db(out + done, chunk);
    done += chunk;
  }
  atomic_store_explicit(&e->last_frames, n, memory_order_relaxed);
  atomic_store_explicit(&e->in_rms_db, in_db, memory_order_relaxed);
  atomic_store_explicit(&e->out_rms_db, out_db, memory_order_relaxed);
}

/* --- latency probe: start, then GCC-PHAT on the capture (non-RT) --- */

es_status es_engine_probe_start(es_engine *e) {
  if (e == NULL) return ES_ERR_ARG;
  if (atomic_load_explicit(&e->probe_state, memory_order_acquire) == PROBE_RUNNING) return ES_ERR_BUSY;
  e->probe_pos = 0;
  e->result_ready = 0;
  atomic_store_explicit(&e->probe_state, PROBE_RUNNING, memory_order_release);
  return ES_OK;
}

/* Plain iterative radix-2 complex FFT, in place. n must be a power of two.
 * inverse = 1 computes the unscaled inverse (we only need the peak position). */
static void stub_fft(double *re, double *im, int n, int inverse) {
  for (int i = 1, j = 0; i < n; i++) { /* bit-reversal permutation */
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      double t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (int len = 2; len <= n; len <<= 1) { /* butterflies */
    double ang = 2.0 * STUB_PI / len * (inverse ? 1.0 : -1.0);
    double wr = cos(ang), wi = sin(ang);
    for (int i = 0; i < n; i += len) {
      double cr = 1.0, ci = 0.0;
      for (int k = 0; k < len / 2; k++) {
        int a = i + k, b = i + k + len / 2;
        double xr = re[b] * cr - im[b] * ci;
        double xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
        double ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/* Median of a small array (sorts it in place, insertion sort). */
static float stub_median(float *v, int n) {
  for (int i = 1; i < n; i++) {
    float key = v[i];
    int j = i - 1;
    while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; j--; }
    v[j + 1] = key;
  }
  if (n == 0) return 0.0f;
  return n % 2 ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

/* GCC-PHAT per sweep (20-engine 12): G = C conj(R) / (|C conj(R)| + 1e-12),
 * g = IFFT(G), peak over lags 0..0.45 s with parabolic interpolation,
 * psr = peak over the largest value outside +/- 2 ms of the peak. */
static es_status stub_probe_compute(es_engine *e, es_probe_result *r) {
  const int N = STUB_PROBE_NFFT;
  const double sr = (double)e->cfg.io_rate;
  const int win = (int)lround(0.75 * sr);     /* capture window per sweep */
  const int max_lag = (int)lround(0.45 * sr);
  const int guard = (int)lround(0.002 * sr);  /* +/- 2 ms around the peak */
  double *rr = (double *)calloc((size_t)N, sizeof(double));
  double *ri = (double *)calloc((size_t)N, sizeof(double));
  double *cr = (double *)calloc((size_t)N, sizeof(double));
  double *ci = (double *)calloc((size_t)N, sizeof(double));
  if (rr == NULL || ri == NULL || cr == NULL || ci == NULL) {
    free(rr); free(ri); free(cr); free(ci);
    return ES_ERR_NOMEM;
  }
  /* reference spectrum R: the sweep, zero padded */
  for (int i = 0; i < e->sweep_len; i++) rr[i] = e->sweep[i];
  stub_fft(rr, ri, N, 0);

  float lags_ms[STUB_PROBE_SWEEPS], psrs[STUB_PROBE_SWEEPS], all_psr[STUB_PROBE_SWEEPS];
  int n_ok = 0;
  for (int k = 0; k < STUB_PROBE_SWEEPS; k++) {
    /* capture window starting at the sweep start, zero padded */
    memset(cr, 0, sizeof(double) * (size_t)N);
    memset(ci, 0, sizeof(double) * (size_t)N);
    int start = k * e->sweep_period;
    for (int i = 0; i < win && start + i < e->cap_len; i++) cr[i] = e->cap[start + i];
    stub_fft(cr, ci, N, 0);
    /* phase transform: keep only the phase of C conj(R) */
    for (int i = 0; i < N; i++) {
      double xr = cr[i] * rr[i] + ci[i] * ri[i];   /* real part of C * conj(R) */
      double xi = ci[i] * rr[i] - cr[i] * ri[i];   /* imaginary part */
      double mag = sqrt(xr * xr + xi * xi) + 1e-12;
      cr[i] = xr / mag;
      ci[i] = xi / mag;
    }
    stub_fft(cr, ci, N, 1);                         /* cr now holds g (unscaled) */

    int peak = 0;
    for (int l = 1; l <= max_lag; l++)
      if (fabs(cr[l]) > fabs(cr[peak])) peak = l;
    double y0 = fabs(cr[peak]);
    double ym = fabs(cr[(peak - 1 + N) % N]);
    double yp = fabs(cr[peak + 1]);
    double denom = ym - 2.0 * y0 + yp;
    double delta = fabs(denom) > 1e-20 ? 0.5 * (ym - yp) / denom : 0.0;
    double side = 0.0;                               /* largest value away from the peak */
    for (int l = 0; l <= max_lag; l++)
      if (abs(l - peak) > guard && fabs(cr[l]) > side) side = fabs(cr[l]);
    float psr = side > 0.0 ? (float)(20.0 * log10(y0 / side)) : 100.0f;
    all_psr[k] = psr;
    if (psr >= 10.0f) {
      lags_ms[n_ok] = (float)((peak + delta) * 1000.0 / sr);
      psrs[n_ok] = psr;
      n_ok++;
    }
  }
  free(rr); free(ri); free(cr); free(ci);

  memset(r, 0, sizeof(*r));
  r->n_ok = n_ok;
  if (n_ok > 0) {
    float lo = lags_ms[0], hi = lags_ms[0];
    for (int i = 1; i < n_ok; i++) {
      if (lags_ms[i] < lo) lo = lags_ms[i];
      if (lags_ms[i] > hi) hi = lags_ms[i];
    }
    r->spread_ms = hi - lo;
    r->loop_ms = stub_median(lags_ms, n_ok);
    r->psr_db = stub_median(psrs, n_ok);
  } else {
    r->psr_db = stub_median(all_psr, STUB_PROBE_SWEEPS);
  }
  return n_ok < 3 ? ES_ERR_PROBE_WEAK : ES_OK;
}

es_status es_engine_probe_result(es_engine *e, es_probe_result *r) {
  if (e == NULL || r == NULL) return ES_ERR_ARG;
  if (atomic_load_explicit(&e->probe_state, memory_order_acquire) != PROBE_CAPTURED) return ES_ERR_STATE;
  if (!e->result_ready) { /* computed once, then cached until the next start */
    e->result_status = stub_probe_compute(e, &e->result);
    if (e->result_status == ES_ERR_NOMEM) return ES_ERR_NOMEM;
    e->result_ready = 1;
  }
  *r = e->result;
  return e->result_status;
}
