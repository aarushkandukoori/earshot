/* test_fuzz.c - one simulated hour of hostile input through the whole engine (20-engine 14.4).
 *
 * One thread alternates es_engine_render (callback sizes uniform 16..1,104 at 48 kHz) and
 * es_engine_run_hops (the manual-worker hook of internal.h), so the run is deterministic and a failing
 * seed replays exactly. While it runs:
 *   - every 0.5 to 5 s the input switches among silence, white noise at -60 to 0 dBFS, full-scale square
 *     waves, DC +1 / -1, denormals (1e-40), NaN bursts, +/-Inf bursts, sweeps and the test-vector mixture;
 *   - every 1 to 10 s it randomizes the params (including out-of-range, NaN and Inf values), toggles
 *     mute, sends cues (also out-of-range beep counts) and swaps each slot's conditioning among the four
 *     test-vector cases (person, everyone, noown, nowespk) and NULL;
 *   - it starts a latency probe 1 to 5 s after each engine starts and then every 20 to 120 s, and
 *     reads the result when the capture is done;
 *   - the engine (3 instances) is rebuilt 6 times per run with a different callback grant and slack
 *     (64/43, 1104/22, 256/64, 480/43, 16/22, 128/43), so the clean path and the underrun path (large
 *     callbacks on a small grant) both get their share of the hour.
 * Checks (one PASS / FAIL line each, exit 1 on any FAIL):
 *   fuzz_output_finite      every output sample is finite
 *   fuzz_output_peak        every |y| <= 0.7079458 + 1e-6 (the -3 dBFS limiter ceiling)
 *   fuzz_run_hops_max       no es_engine_run_hops call returned more than 64 hops
 *   fuzz_hops_keep_up       after every run_hops, stats.hops == input_position / 64: every complete hop
 *                           ran and none was lost, so the engine keeps producing hops
 *   fuzz_hops_total         per engine, hops == rendered 48 kHz samples / 192 within 1 hop
 *   fuzz_set_cond_status    every set_cond returned ES_OK or ES_ERR_BUSY
 *   fuzz_probe_results      every probe started, and finished with ES_OK or ES_ERR_PROBE_WEAK and finite fields
 * INFO lines report nan_resets, underruns, slack, howl, BUSY counts, probes and per-hop timings.
 *
 * Usage: FUZZ_SECONDS=3600 FUZZ_SEED=20261107 build/test_fuzz [--export DIR]
 *   make fuzz runs the hour under nice; make test runs FUZZ_SECONDS=60. The default model is synthetic
 *   (es_model_synthetic) with conditioning enrolled from synthetic voices; --export DIR loads
 *   DIR/earshot.esw, the four reference conditionings of DIR/testvec_short.est and the mixture `in` of
 *   DIR/testvec_long.est (or of the short file).
 * Threads: a single-threaded test program (the main thread plays both the audio thread and the worker).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "internal.h"

#define SR        48000
#define CEILING   (0.7079458f + 1e-6f)
#define N_SEG     6
#define MAX_CB    1104

/* ------------------------------------------------------------------------------------------------
 * Deterministic random numbers (xorshift64*), seeded from FUZZ_SEED.
 * ---------------------------------------------------------------------------------------------- */
static uint64_t g_s = 20261107u;
static uint32_t rnd(void) {
  g_s ^= g_s >> 12;
  g_s ^= g_s << 25;
  g_s ^= g_s >> 27;
  return (uint32_t)((g_s * 2685821657736338717ULL) >> 32);
}
static double unif(double lo, double hi) { return lo + (hi - lo) * (rnd() / 4294967296.0); }
static float noise1(void) { return (float)unif(-1.0, 1.0); }

/* ------------------------------------------------------------------------------------------------
 * Model, the four conditioning cases and the mixture.
 * ---------------------------------------------------------------------------------------------- */
static const char *CASE_NAMES[4] = { "person", "everyone", "noown", "nowespk" };

typedef struct {
  es_model *m;
  es_cond *cases[4];
  float *mix16;        /* test-vector mixture at 16 kHz */
  int mix_len;
} assets;

/* A synthetic "voice" at 16 kHz: 10 harmonics of f0 with vibrato, a 4 Hz syllable envelope, a bit of
   noise. Good enough to enroll and to look like speech to the gate. */
static void synth_voice(float *x, int n, double f0, double env_phase, float level) {
  double ph = 0.0;
  for (int i = 0; i < n; i++) {
    double t = i / 16000.0;
    ph += 2.0 * ES_PI * f0 * (1.0 + 0.04 * sin(2.0 * ES_PI * 3.0 * t)) / 16000.0;
    double env = 0.5 + 0.5 * sin(2.0 * ES_PI * 4.0 * t + env_phase);
    double s = 0.0;
    for (int h = 1; h <= 10; h++) s += sin(h * ph) / h;
    x[i] = level * (float)(env * env * s) + 0.003f * noise1();
  }
}

static int load_synthetic(assets *a, uint32_t seed) {
  if (es_model_synthetic(seed, 0, &a->m) != ES_OK) return -1;
  enum { NE = 96000 };                               /* 6 s enrollments, 188 frames */
  static float pt[NE], po[NE];
  static uint8_t ones[200];
  memset(ones, 1, sizeof(ones));
  synth_voice(pt, NE, 130.0, 0.0, 0.1f);
  synth_voice(po, NE, 210.0, 1.0, 0.1f);
  float e_t[ES_WESPK_DIM], e_o[ES_WESPK_DIM];
  for (int i = 0; i < ES_WESPK_DIM; i++) { e_t[i] = 0.06f * noise1(); e_o[i] = 0.06f * noise1(); }
  es_voiceprint *vt = es_voiceprint_new(), *vo = es_voiceprint_new();
  if (!vt || !vo || es_enroll(a->m, pt, NE, ones, ES_ROLE_TARGET, e_t, vt) != ES_OK ||
      es_enroll(a->m, po, NE, ones, ES_ROLE_OWN, e_o, vo) != ES_OK)
    return -1;
  /* The four cases of 20-engine 7.4: person, everyone, noown, nowespk. */
  const es_mode mode[4] = { ES_MODE_PERSON, ES_MODE_EVERYONE, ES_MODE_PERSON, ES_MODE_PERSON };
  const es_voiceprint *tg[4] = { vt, NULL, vt, vt }, *ow[4] = { vo, vo, NULL, vo };
  const int wes[4] = { 1, 1, 1, 0 };
  for (int k = 0; k < 4; k++) {
    a->cases[k] = es_cond_new();
    if (!a->cases[k] || es_cond_build(a->m, mode[k], tg[k], ow[k], wes[k], a->cases[k]) != ES_OK) return -1;
  }
  es_voiceprint_free(vt);
  es_voiceprint_free(vo);
  /* Mixture: two voices plus noise, 10 s. */
  a->mix_len = 160000;
  a->mix16 = malloc(sizeof(float) * (size_t)a->mix_len);
  float *b = malloc(sizeof(float) * (size_t)a->mix_len);
  if (!a->mix16 || !b) { free(b); return -1; }
  synth_voice(a->mix16, a->mix_len, 130.0, 0.5, 0.1f);
  synth_voice(b, a->mix_len, 180.0, 2.0, 0.06f);
  for (int i = 0; i < a->mix_len; i++) a->mix16[i] += b[i];
  free(b);
  return 0;
}

/* Reads the `in` array of a test-vector file into a->mix16. 0 on success. */
static int load_mix(assets *a, const char *path) {
  void *buf = NULL;
  size_t n = 0;
  es_container ct;
  char err[128];
  if (es_read_file(path, &buf, &n) != ES_OK) return -1;
  int ok = es_container_open(buf, n, "EST1", &ct, err, sizeof(err)) == ES_OK;
  if (ok) {
    const es_array *in = es_container_find(&ct, "in");
    ok = in && in->dtype == ES_DT_F32 && in->ndim == 1;
    if (ok) {
      a->mix_len = in->shape[0];
      a->mix16 = es_tv_f32(&ct, "in", (size_t)a->mix_len);
      ok = a->mix16 != NULL;
    }
    es_container_close(&ct);
  }
  free(buf);
  return ok ? 0 : -1;
}

static int load_export(assets *a, const char *dir) {
  char path[1024], err[256];
  snprintf(path, sizeof(path), "%s/earshot.esw", dir);
  if (es_model_load_file(path, &a->m, err, sizeof(err)) != ES_OK) {
    fprintf(stderr, "test_fuzz: cannot load %s: %s\n", path, err);
    return -1;
  }
  snprintf(path, sizeof(path), "%s/testvec_short.est", dir);
  void *buf = NULL;
  size_t n = 0;
  es_container ct;
  if (es_read_file(path, &buf, &n) != ES_OK || es_container_open(buf, n, "EST1", &ct, err, sizeof(err)) != ES_OK) {
    fprintf(stderr, "test_fuzz: cannot open %s\n", path);
    free(buf);
    return -1;
  }
  int ok = 1;
  for (int k = 0; k < 4 && ok; k++) {                /* the reference cond.<case>.* arrays */
    char name[64];
    a->cases[k] = es_cond_new();
    snprintf(name, sizeof(name), "cond.%s.film", CASE_NAMES[k]);
    float *film = es_tv_f32(&ct, name, 4096);
    snprintf(name, sizeof(name), "cond.%s.k", CASE_NAMES[k]);
    float *kk = es_tv_f32(&ct, name, 4 * 68 * 16);
    snprintf(name, sizeof(name), "cond.%s.v", CASE_NAMES[k]);
    float *vv = es_tv_f32(&ct, name, 4 * 68 * 16);
    snprintf(name, sizeof(name), "cond.%s.n", CASE_NAMES[k]);
    const es_array *nn = es_container_find(&ct, name);
    ok = a->cases[k] && film && kk && vv && nn && nn->dtype == ES_DT_I32 && nn->nbytes == 4;
    if (ok) {
      memcpy(a->cases[k]->film, film, sizeof(float) * 4096);
      memcpy(a->cases[k]->k, kk, sizeof(float) * 4 * 68 * 16);
      memcpy(a->cases[k]->v, vv, sizeof(float) * 4 * 68 * 16);
      memcpy(&a->cases[k]->n_tokens, nn->data, 4);
      a->cases[k]->mode = k == 1 ? ES_MODE_EVERYONE : ES_MODE_PERSON;
    }
    free(film);
    free(kk);
    free(vv);
  }
  es_container_close(&ct);
  free(buf);
  if (!ok) return -1;
  snprintf(path, sizeof(path), "%s/testvec_long.est", dir);
  if (load_mix(a, path) != 0) {
    snprintf(path, sizeof(path), "%s/testvec_short.est", dir);
    if (load_mix(a, path) != 0) return -1;
  }
  return 0;
}

/* ------------------------------------------------------------------------------------------------
 * Hostile input generator at 48 kHz. gen_start picks the parameters of a new segment, gen_fill
 * writes the next n samples.
 * ---------------------------------------------------------------------------------------------- */
enum { IN_SILENCE, IN_NOISE, IN_SQUARE, IN_DC, IN_DENORMAL, IN_NAN, IN_INF, IN_SWEEP, IN_MIX, IN_KINDS };
static const char *IN_NAMES[IN_KINDS] = { "silence", "noise", "square", "dc", "denormal", "nan_burst",
                                          "inf_burst", "sweep", "mixture" };

typedef struct {
  int kind;
  float amp;              /* noise, burst background, sweep and mixture level */
  double phase, dphase;   /* square / sweep phase in cycles and its step per sample */
  double sweep_mul;       /* sweep: dphase is multiplied by this every sample (exponential sweep) */
  float dc;
  int burst;              /* samples left in the current NaN / Inf burst */
  double mixpos;          /* mixture read position in 16 kHz samples */
  const assets *a;
} gen;

static void gen_start(gen *g, int kind, int64_t seg_len) {
  g->kind = kind;
  g->burst = 0;
  g->phase = 0.0;
  switch (kind) {
    case IN_NOISE: g->amp = (float)pow(10.0, unif(-60.0, 0.0) / 20.0); break;
    case IN_SQUARE: g->amp = 1.0f; g->dphase = unif(20.0, 8000.0) / SR; break;
    case IN_DC: g->dc = (rnd() & 1) ? 1.0f : -1.0f; break;
    case IN_NAN: case IN_INF: g->amp = (float)pow(10.0, unif(-40.0, 0.0) / 20.0); break;
    case IN_SWEEP: {
      double f0 = 20.0, f1 = 20000.0;
      if (rnd() & 1) { f0 = 20000.0; f1 = 20.0; }   /* down sweeps too */
      g->dphase = f0 / SR;
      g->sweep_mul = pow(f1 / f0, 1.0 / (double)(seg_len > 1 ? seg_len : 1));
      g->amp = (float)pow(10.0, unif(-40.0, 0.0) / 20.0);
      break;
    }
    case IN_MIX:
      g->mixpos = unif(0.0, g->a->mix_len - 1);
      g->amp = (float)pow(10.0, unif(-20.0, 12.0) / 20.0);   /* up to +12 dB: overdrives the clamp */
      break;
    default: break;
  }
}

static void gen_fill(gen *g, float *x, int n) {
  for (int i = 0; i < n; i++) {
    float v = 0.0f;
    switch (g->kind) {
      case IN_NOISE: v = g->amp * noise1(); break;
      case IN_SQUARE:
        v = g->phase < 0.5 ? 1.0f : -1.0f;
        g->phase += g->dphase;
        if (g->phase >= 1.0) g->phase -= 1.0;
        break;
      case IN_DC: v = g->dc; break;
      case IN_DENORMAL: v = (rnd() & 1) ? 1e-40f : -1e-40f; break;
      case IN_NAN: case IN_INF:
        if (g->burst > 0) {
          g->burst--;
          v = g->kind == IN_NAN ? NAN : ((rnd() & 1) ? INFINITY : -INFINITY);
        } else {
          v = g->amp * noise1();
          if (rnd() % 4000 == 0) g->burst = 1 + (int)(rnd() % 2000);   /* bursts of 1..2,000 samples */
        }
        break;
      case IN_SWEEP:
        v = g->amp * (float)sin(2.0 * ES_PI * g->phase);
        g->phase += g->dphase;
        if (g->phase >= 1.0) g->phase -= 1.0;
        g->dphase *= g->sweep_mul;
        break;
      case IN_MIX: {                                 /* 16 kHz mixture, linear interpolation to 48 kHz */
        int j = (int)g->mixpos;
        double fr = g->mixpos - j;
        int j1 = j + 1 < g->a->mix_len ? j + 1 : 0;
        v = g->amp * (float)((1.0 - fr) * g->a->mix16[j] + fr * g->a->mix16[j1]);
        g->mixpos += 1.0 / 3.0;
        if (g->mixpos >= g->a->mix_len) g->mixpos -= g->a->mix_len;
        break;
      }
      default: break;                                /* silence */
    }
    x[i] = v;
  }
}

/* ------------------------------------------------------------------------------------------------
 * Control changes: params (often out of range or non-finite), mute, cues, conditioning swaps.
 * ---------------------------------------------------------------------------------------------- */
static float rand_field(float lo, float hi) {
  uint32_t r = rnd() % 100;
  if (r < 8) return NAN;
  if (r < 12) return (rnd() & 1) ? INFINITY : -INFINITY;
  if (r < 30) return (float)unif(-1000.0, 1000.0);   /* mostly out of range */
  return (float)unif(lo, hi);
}
static int rand_flag(void) {
  uint32_t r = rnd() % 10;
  return r < 2 ? (int)(rnd() % 11) - 5 : (int)(r & 1);   /* sometimes not 0 / 1 at all */
}

typedef struct {
  uint64_t set_cond_calls, set_cond_busy, set_cond_bad, cues, mutes;
} ctl_counts;

static void randomize(es_engine *e, const assets *a, int *muted, ctl_counts *cc) {
  es_params p;
  es_params_default(&p);
  p.voice_db = rand_field(-10.0f, 6.0f);
  static const float rooms[4] = { -60.0f, -24.0f, -15.0f, -8.0f };
  p.room_db = (rnd() & 1) ? rooms[rnd() % 4] : rand_field(-60.0f, -8.0f);
  p.max_gain_db = rand_field(0.0f, 18.0f);
  /* Loud settings often (trim 0 dB half the time, bypass asked for a third of the time), so the -3 dBFS
     bound is really pushed: full-scale squares and DC through bypass or the room path must hit the
     limiter. A control build without the limiter fails fuzz_output_peak (4 of 4 seeds at 60 s). */
  p.out_gain_db = (rnd() & 1) ? 0.0f : rand_field(-40.0f, 0.0f);
  for (int s = 0; s < ES_MAX_INSTANCES; s++) p.slot_gain[s] = rand_field(0.0f, 1.0f);
  p.xfade_ms = rand_field(4.0f, 500.0f);
  p.duck_own = rand_flag();
  p.gate = rand_flag();
  p.agc = rand_flag();
  p.bypass = rnd() % 3 == 0 ? rand_flag() : 0;
  es_engine_set_params(e, &p);
  if (*muted ? unif(0, 1) < 0.7 : unif(0, 1) < 0.2) {     /* mostly unmuted, so the output is exercised */
    *muted = !*muted;
    es_engine_set_mute(e, *muted);
    cc->mutes++;
  }
  if (rnd() % 10 < 4) {
    es_engine_cue(e, (int)(rnd() % 7) - 1);                /* -1..5 beeps: clamped to 1..3 */
    cc->cues++;
  }
  for (int s = 0; s < ES_MAX_INSTANCES; s++) {
    if (rnd() & 1) continue;
    int k = (int)(rnd() % 5);                              /* 4 = NULL (empty slot) */
    es_status st = es_engine_set_cond(e, s, k < 4 ? a->cases[k] : NULL, (int)(rnd() & 1));
    cc->set_cond_calls++;
    if (st == ES_ERR_BUSY) cc->set_cond_busy++;
    else if (st != ES_OK) cc->set_cond_bad++;
  }
}

/* ------------------------------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------------------------------- */
static int g_pass = 0, g_fail = 0;
static void check(const char *name, int ok, double value, double limit) {
  printf("%s %s %.9g %.9g\n", ok ? "PASS" : "FAIL", name, value, limit);
  if (ok) g_pass++; else g_fail++;
}

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint64_t g_hop_us[65536];                     /* histogram of per-hop compute time (us) */

int main(int argc, char **argv) {
  const char *dir = NULL;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--export") == 0 && i + 1 < argc) dir = argv[++i];
    else { fprintf(stderr, "usage: FUZZ_SECONDS=3600 FUZZ_SEED=n test_fuzz [--export DIR]\n"); return 2; }
  }
  const char *es = getenv("FUZZ_SECONDS"), *ss = getenv("FUZZ_SEED");
  double seconds = es ? atof(es) : 3600.0;
  if (!(seconds >= 6.0)) seconds = 6.0;
  uint64_t seed = ss ? strtoull(ss, NULL, 10) : 20261107u;
  g_s = seed ? seed : 1;
  printf("test_fuzz (20-engine 14.4): seed %llu, %.0f simulated seconds at 48 kHz, model %s\n",
         (unsigned long long)seed, seconds, dir ? dir : "synthetic");
  fflush(stdout);

  assets a;
  memset(&a, 0, sizeof(a));
  int lr = dir ? load_export(&a, dir) : load_synthetic(&a, (uint32_t)(seed * 2654435761u) | 1u);
  if (lr != 0) { printf("FAIL fuzz_setup 0 1\n"); return 1; }
  printf("INFO cases n_tokens person %d everyone %d noown %d nowespk %d, mixture %d samples at 16 kHz\n",
         a.cases[0]->n_tokens, a.cases[1]->n_tokens, a.cases[2]->n_tokens, a.cases[3]->n_tokens, a.mix_len);

  static const int seg_frames[N_SEG] = { 64, 1104, 256, 480, 16, 128 };
  static const int seg_slack[N_SEG] = { 43, 22, 64, 43, 22, 43 };
  const int64_t total = (int64_t)(seconds * SR), seg_len = total / N_SEG;
  static float in[MAX_CB], out[MAX_CB];
  gen g;
  memset(&g, 0, sizeof(g));
  g.a = &a;
  ctl_counts cc;
  memset(&cc, 0, sizeof(cc));
  uint64_t callbacks = 0, nonfinite = 0, keepup_bad = 0, hops_all = 0, nan_all = 0, under_all = 0, howl_cb = 0;
  uint64_t kinds[IN_KINDS] = { 0 }, probes_started = 0, probes_ok = 0, probes_weak = 0, probes_bad = 0;
  float peak = 0.0f;
  int max_h = 0, total_bad = 0;
  double t_start = now_s(), next_report = 300.0;
  int64_t t = 0;                                      /* I/O samples rendered over the whole run */

  for (int seg = 0; seg < N_SEG; seg++) {
    es_engine_config cfg = { SR, seg_frames[seg], 4096, ES_MAX_INSTANCES, seg_slack[seg] };
    es_engine *e = NULL;
    if (es_engine_new(a.m, &cfg, &e) != ES_OK) { printf("FAIL fuzz_engine_new 0 1\n"); return 1; }
    es_engine_set_cond(e, 0, a.cases[0], 1);          /* before manual mode: picked up at once */
    es_engine_set_cond(e, 1, a.cases[1], 1);
    es_engine_set_manual_worker(e, 1);
    es_engine_set_mute(e, 0);
    int muted = 0, probe_pending = 0;
    uint64_t seg_hops = 0;
    int64_t seg_rendered = 0, seg_end = (seg == N_SEG - 1) ? total : t + seg_len;
    int64_t next_input = t, next_ctl = t + (int64_t)(unif(1.0, 10.0) * SR);
    /* The first probe of each engine starts 1 to 5 s in (so the 60 s make-test run covers probes too),
       later ones every 20 to 120 s. */
    int64_t next_probe = t + (int64_t)(unif(1.0, 5.0) * SR), next_poll = 0, probe_t0 = 0;
    while (t < seg_end) {
      int n = 16 + (int)(rnd() % (MAX_CB - 16 + 1));
      if (t >= next_input) {                          /* new input segment every 0.5 to 5 s */
        int64_t len = (int64_t)(unif(0.5, 5.0) * SR);
        int kind = (int)(rnd() % IN_KINDS);
        gen_start(&g, kind, len);
        kinds[kind]++;
        next_input = t + len;
      }
      if (t >= next_ctl) {                            /* control changes every 1 to 10 s */
        randomize(e, &a, &muted, &cc);
        next_ctl = t + (int64_t)(unif(1.0, 10.0) * SR);
      }
      if (!probe_pending && t >= next_probe) {        /* a latency probe every 20 to 120 s */
        if (es_engine_probe_start(e) != ES_OK) probes_bad++;
        probes_started++;
        probe_pending = 1;
        probe_t0 = t;
        next_poll = t + SR / 4;
      }
      if (probe_pending && t >= next_poll) {
        es_probe_result r;
        es_status ps = es_engine_probe_result(e, &r);
        if (ps == ES_ERR_STATE && t - probe_t0 < 10LL * SR) {
          next_poll = t + SR / 4;
        } else {
          int fields = isfinite(r.loop_ms) && isfinite(r.spread_ms) && isfinite(r.psr_db) && r.spread_ms >= 0.0f &&
                       r.n_ok >= 0 && r.n_ok <= 5;
          if (ps == ES_OK && fields) probes_ok++;
          else if (ps == ES_ERR_PROBE_WEAK && fields) probes_weak++;
          else probes_bad++;                          /* also a probe stuck for 10 s */
          probe_pending = 0;
          next_probe = t + (int64_t)(unif(20.0, 120.0) * SR);
        }
      }

      gen_fill(&g, in, n);
      es_engine_render(e, in, out, n);
      for (int i = 0; i < n; i++) {
        float v = out[i];
        if (!isfinite(v)) nonfinite++;
        else if (fabsf(v) > peak) peak = fabsf(v);
      }
      int h = es_engine_run_hops(e);
      if (h > max_h) max_h = h;
      if (h < 0) total_bad++;
      seg_hops += (uint64_t)(h > 0 ? h : 0);
      es_stats st;
      es_engine_get_stats(e, &st);
      if (st.hops != seg_hops || st.hops != es_engine_input_position(e) / ES_HOP) keepup_bad++;
      if (st.howl) howl_cb++;
      uint16_t tb[ES_TIMINGS_SIZE];
      int got = es_engine_read_timings(e, tb, ES_TIMINGS_SIZE);
      for (int i = 0; i < got; i++) g_hop_us[tb[i]]++;
      t += n;
      seg_rendered += n;
      callbacks++;
      if (t >= next_report * SR) {
        printf("INFO progress %.0f / %.0f s, hops %llu, wall %.0f s\n", next_report, seconds,
               (unsigned long long)(hops_all + seg_hops), now_s() - t_start);
        fflush(stdout);
        next_report += 300.0;
      }
    }
    es_stats st;
    es_engine_get_stats(e, &st);
    int64_t want = seg_rendered / (3 * ES_HOP);
    int64_t diff = (int64_t)st.hops - want;
    if (diff < -1 || diff > 1) total_bad++;
    printf("INFO engine %d: frames %d slack %d -> %d, rendered %.1f s, hops %llu (want %lld), underruns %llu, "
           "nan_resets %llu, probe_runs %llu, compute_us_max %.0f\n",
           seg, seg_frames[seg], seg_slack[seg], st.slack16, seg_rendered / (double)SR,
           (unsigned long long)st.hops, (long long)want, (unsigned long long)st.underruns,
           (unsigned long long)st.nan_resets, (unsigned long long)st.probe_runs, st.compute_us_max);
    if (probe_pending) probes_started--;              /* cut off by the end of the segment, not judged */
    hops_all += st.hops;
    nan_all += st.nan_resets;
    under_all += st.underruns;
    es_engine_free(e);
  }

  /* Report. */
  uint64_t n_t = 0, acc = 0;
  for (int i = 0; i < 65536; i++) n_t += g_hop_us[i];
  int p50 = -1, p99 = -1, p999 = -1, pmax = 0;
  for (int i = 0; i < 65536; i++) {
    acc += g_hop_us[i];
    if (g_hop_us[i]) pmax = i;
    if (p50 < 0 && acc * 2 >= n_t) p50 = i;
    if (p99 < 0 && acc * 100 >= n_t * 99) p99 = i;
    if (p999 < 0 && acc * 1000 >= n_t * 999) p999 = i;
  }
  printf("INFO seed %llu, %.0f s simulated in %.1f s wall, %llu callbacks, %llu hops, nan_resets %llu, "
         "underruns %llu, callbacks with howl %llu\n",
         (unsigned long long)seed, t / (double)SR, now_s() - t_start, (unsigned long long)callbacks,
         (unsigned long long)hops_all, (unsigned long long)nan_all, (unsigned long long)under_all,
         (unsigned long long)howl_cb);
  printf("INFO set_cond calls %llu (busy %llu), cues %llu, mute toggles %llu; probes started %llu, ok %llu, weak %llu\n",
         (unsigned long long)cc.set_cond_calls, (unsigned long long)cc.set_cond_busy, (unsigned long long)cc.cues,
         (unsigned long long)cc.mutes, (unsigned long long)probes_started, (unsigned long long)probes_ok,
         (unsigned long long)probes_weak);
  printf("INFO input segments:");
  for (int k = 0; k < IN_KINDS; k++) printf(" %s %llu", IN_NAMES[k], (unsigned long long)kinds[k]);
  printf("\nINFO hop compute us (3 instances, this thread, no FTZ): p50 %d p99 %d p99.9 %d max %d over %llu hops\n",
         p50, p99, p999, pmax, (unsigned long long)n_t);
  check("fuzz_output_finite", nonfinite == 0, (double)nonfinite, 0);
  check("fuzz_output_peak", peak <= CEILING, peak, CEILING);
  check("fuzz_run_hops_max", max_h <= ES_MAX_HOPS_CALL, max_h, ES_MAX_HOPS_CALL);
  check("fuzz_hops_keep_up", keepup_bad == 0, (double)keepup_bad, 0);
  check("fuzz_hops_total", total_bad == 0 && hops_all > 0, total_bad, 0);
  check("fuzz_set_cond_status", cc.set_cond_bad == 0, (double)cc.set_cond_bad, 0);
  check("fuzz_probe_results", probes_bad == 0 && probes_ok + probes_weak == probes_started, (double)probes_bad, 0);
  printf("test_fuzz: %d passed, %d failed\n", g_pass, g_fail);
  for (int k = 0; k < 4; k++) es_cond_free(a.cases[k]);
  free(a.mix16);
  es_model_free(a.m);
  return g_fail ? 1 : 0;
}
