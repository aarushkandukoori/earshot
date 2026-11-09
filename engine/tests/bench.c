/* bench.c - per-hop and per-callback timing of the engine (20-engine 14.6, contract 11 "hop p50 <= 250 us").
 *
 * Usage (the Makefile sets VECLIB_MAXIMUM_THREADS=1):
 *   build/bench [--ecore] [--note TEXT] [EXPORT_DIR | synthetic]
 * EXPORT_DIR (for example ../ml/exports/random_v1) loads EXPORT_DIR/earshot.esw and uses its test vectors;
 * a missing directory or the word "synthetic" uses es_model_synthetic (no export needed).
 * --ecore re-runs the whole bench under `taskpolicy -c background` (the efficiency cores); its rows get
 * the suffix "_ecore". Tails on the E-cores are not meaningful (docs/designs/product.md B.2).
 *
 * Thread: everything runs on the main thread, default QoS (a performance core), single-threaded BLAS
 * (BLASSetThreading plus VECLIB_MAXIMUM_THREADS=1) and FPCR.FZ set, like the real worker (20-engine 3.6).
 *
 * Rows (each one JSON line on stdout; a human summary goes to stderr):
 *   hop_1inst    es_stream_hop on one stream, the test-vector mixture (synthetic: white noise),
 *                500 warm-up hops, then 20,000 hops timed one by one with mach_absolute_time
 *   hop_3inst    the same with 3 streams; one sample = the 3 hops of one 4 ms step (the worker's cost)
 *   render_64    48 kHz engine, 3 instances, 64-frame callbacks: es_engine_render plus the
 *                es_engine_run_hops it triggers, timed per callback (1,000 warm-up, 20,000 timed)
 *   render_1104  the same with 1,104-frame callbacks (100 warm-up, 2,000 timed: each one runs 5 or 6
 *                steps of 3 hops)
 *   enroll_10s   es_enroll on 10 s of audio with every frame kept (all-ones mask), 1 + 20 runs
 *   cond_build   es_cond_build(PERSON, target, own), 1 + 200 runs
 *   selftest     es_selftest on testvec_short.est, 1 + 10 runs (export only: a synthetic model has no
 *                test vectors)
 * Row fields: name, n, p50_us, p90_us, p99_us, p999_us, max_us (nearest-rank percentiles), load_start
 * (1-minute load average when the row starts), model ("synthetic" or the export folder name), hop_path
 * ("sgemv" = the default build and what the app ships, "xt" = a -DES_HOP_SGEMM_XT build; es_hop_path in
 * nn.c), and note when --note is given.
 * Deviations from the 14.6 field list (E-3 fix pass): hop_path is added, because the two hop paths differ
 * by about 80 us per hop; p999_us is null when n < 1000 (with fewer samples it is just the maximum).
 * Rows E-3 wrote before the field existed were tagged afterwards from their notes: "xt_unsplit" (E-3's
 * one transposed sgemm per product, which allocates on the iOS simulator) or "sgemv".
 * --note text may not contain a double quote, a backslash or a control character (it is printed into
 * JSON as is, so bench refuses such a note instead of writing a broken line). Conditioning uses 48 target + 16 own tokens (all-ones enrollment masks), the
 * largest token count, so the attention cost is the worst case.
 * Gate (target, printed on stderr, never the exit code): hop_1inst p50 <= 250 us.
 */
#include <libgen.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mach/mach_time.h>

#include "internal.h"

static double g_tick_us;              /* microseconds per mach_absolute_time tick */
static const char *g_model = "synthetic", *g_note = NULL, *g_suffix = "";
static int g_failed = 0;              /* a row could not run: main exits 1 (bench-save then keeps no rows) */

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : (x > y ? 1 : 0);
}
/* Nearest-rank percentile of a sorted array: the smallest value with at least p of the samples at or
   below it. */
static double pct(const double *sorted, int n, double p) {
  int i = (int)ceil(p * n) - 1;
  return sorted[i < 0 ? 0 : (i >= n ? n - 1 : i)];
}
static double now_us(void) { return (double)mach_absolute_time() * g_tick_us; }
static double load1(void) {
  double l[3] = { 0, 0, 0 };
  return getloadavg(l, 3) > 0 ? l[0] : -1.0;
}

/* Sorts t[0..n) and prints the row. */
static void emit(const char *name, double *t, int n, double load_start) {
  qsort(t, (size_t)n, sizeof(double), cmp_double);
  printf("{\"name\": \"%s%s\", \"n\": %d, \"p50_us\": %.1f, \"p90_us\": %.1f, \"p99_us\": %.1f, ", name, g_suffix, n,
         pct(t, n, 0.5), pct(t, n, 0.9), pct(t, n, 0.99));
  if (n >= 1000) printf("\"p999_us\": %.1f, ", pct(t, n, 0.999));
  else printf("\"p999_us\": null, ");                     /* fewer than 1,000 samples: p999 = max */
  printf("\"max_us\": %.1f, \"load_start\": %.2f, \"model\": \"%s\", \"hop_path\": \"%s\"", t[n - 1], load_start, g_model,
         es_hop_path);
  if (g_note) printf(", \"note\": \"%s\"", g_note);
  printf("}\n");
  fflush(stdout);
  fprintf(stderr, "%s%-*s n %6d  p50 %9.1f  p99 %9.1f  max %9.1f us  (load %.2f)\n", name,
          (int)(18 - strlen(name)), g_suffix, n, pct(t, n, 0.5), pct(t, n, 0.99), t[n - 1], load_start);
}

/* FPCR.FZ on this thread, as the worker does (20-engine 3.6). */
static void flush_denormals(void) {
#if defined(__aarch64__)
  uint64_t r;
  __asm__ volatile("mrs %0, fpcr" : "=r"(r));
  r |= 1ull << 24;
  __asm__ volatile("msr fpcr, %0" ::"r"(r));
#endif
}

static uint32_t g_rng = 20261108u;
static float urand(void) {        /* uniform in [-1, 1), xorshift32 */
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return (float)((g_rng >> 8) * (1.0 / 16777216.0)) * 2.0f - 1.0f;
}

/* Everything the rows share, prepared once by main (non-RT, may allocate). */
typedef struct {
  const es_model *m;
  float *mix;                   /* 16 kHz input: testvec_long `in` (10 s) or white noise (10 s) */
  int n_mix;
  float *enr_t, *enr_o;         /* enrollment audio: testvec_short enr_t / enr_o, or noise */
  int Lt, Lo;
  float *e_t, *e_o;             /* WeSpeaker vectors from the test vector, or NULL */
  es_voiceprint *vt, *vo;       /* enrolled with all-ones masks: 48 target and 16 own tokens */
  es_cond *c;                   /* PERSON conditioning built from vt and vo */
  void *est;                    /* testvec_short.est bytes (export only) */
  size_t est_n;
} bench_in;

/* hop_1inst / hop_3inst: n_inst streams, one sample = one 4 ms step of all of them. */
static void row_hops(const bench_in *b, int n_inst, const char *name) {
  enum { WARM = 500, N = 20000 };
  es_stream *st[3] = { NULL, NULL, NULL };
  double *t = malloc(sizeof(double) * N);
  for (int k = 0; k < n_inst; k++) st[k] = es_stream_new(b->m);
  for (int k = 0; k < n_inst; k++)
    if (!st[k] || !t) {                                  /* out of memory: no row (main exits 1) */
      fprintf(stderr, "%s: es_stream_new or malloc failed\n", name);
      for (int q = 0; q < n_inst; q++) es_stream_free(st[q]);
      free(t);
      g_failed = 1;
      return;
    }
  float out[ES_HOP], heads[2];
  double load = load1();
  int pos = 0;
  for (int i = -WARM; i < N; i++) {
    const float *x = b->mix + pos;
    pos += ES_HOP;
    if (pos + ES_HOP > b->n_mix) pos = 0;                  /* loop over the mixture */
    double t0 = now_us();
    for (int k = 0; k < n_inst; k++) es_stream_hop(st[k], b->c, x, out, heads);
    double t1 = now_us();
    if (i >= 0) t[i] = t1 - t0;
  }
  emit(name, t, N, load);
  if (strcmp(name, "hop_1inst") == 0)
    fprintf(stderr, "GATE hop_1inst%s p50 %.1f us, target <= 250 us: %s (hop path %s; target only, shared machine)\n",
            g_suffix, pct(t, N, 0.5), pct(t, N, 0.5) <= 250.0 ? "met" : "NOT met", es_hop_path);
  for (int k = 0; k < n_inst; k++) es_stream_free(st[k]);
  free(t);
}

/* render_64 / render_1104: one callback = es_engine_render plus the es_engine_run_hops it triggers.
   48 kHz, 3 instances (all slots filled, all audible), room sound on, the default gate, duck and AGC.
   The 16 kHz mixture is repeated 3 times per sample to make 48 kHz input. */
static void row_render(const bench_in *b, int frames, int warm, int n, const char *name) {
  es_engine_config cfg = { 48000, frames, 4096, 3, 43 };
  es_engine *e = NULL;
  if (es_engine_new(b->m, &cfg, &e) != ES_OK) {
    fprintf(stderr, "%s: es_engine_new failed\n", name);
    g_failed = 1;
    return;
  }
  for (int s = 0; s < 3; s++) es_engine_set_cond(e, s, b->c, 1);   /* before manual mode: applied at once */
  es_engine_set_manual_worker(e, 1);
  es_params p;
  es_params_default(&p);
  p.slot_gain[1] = p.slot_gain[2] = 0.5f;
  p.room_db = -24.0f;
  es_engine_set_params(e, &p);
  es_engine_set_mute(e, 0);
  float *in = malloc(sizeof(float) * (size_t)frames), *out = malloc(sizeof(float) * (size_t)frames);
  double *t = malloc(sizeof(double) * (size_t)n);
  if (!in || !out || !t) {
    fprintf(stderr, "%s: malloc failed\n", name);
    es_engine_free(e);
    free(in); free(out); free(t);
    g_failed = 1;
    return;
  }
  double load = load1();
  long pos48 = 0;
  for (int i = -warm; i < n; i++) {
    for (int j = 0; j < frames; j++, pos48++) in[j] = b->mix[(pos48 / 3) % b->n_mix];
    double t0 = now_us();
    es_engine_render(e, in, out, frames);
    es_engine_run_hops(e);
    double t1 = now_us();
    if (i >= 0) t[i] = t1 - t0;
  }
  emit(name, t, n, load);
  es_stats s;
  es_engine_get_stats(e, &s);
  fprintf(stderr, "    (%s: %llu hops, %llu underruns, %llu NaN resets)\n", name, (unsigned long long)s.hops,
          (unsigned long long)s.underruns, (unsigned long long)s.nan_resets);
  es_engine_free(e);
  free(in);
  free(out);
  free(t);
}

/* enroll_10s, cond_build, selftest: one-shot calls, repeated (the first run is a warm-up). */
static void row_oneshot(const bench_in *b, int which, int n, const char *name) {
  double *t = malloc(sizeof(double) * (size_t)n);
  int F = es_enroll_num_frames(160000);
  uint8_t *ones = malloc((size_t)F);
  memset(ones, 1, (size_t)F);
  es_voiceprint *v = es_voiceprint_new();
  es_cond *c = es_cond_new();
  if (!t || !ones || !v || !c) {
    fprintf(stderr, "%s: allocation failed\n", name);
    es_voiceprint_free(v); es_cond_free(c); free(ones); free(t);
    g_failed = 1;
    return;
  }
  int use_wespk = b->e_t && b->e_o;
  double load = load1();
  es_status st = ES_OK;
  for (int i = -1; i < n; i++) {
    float err = 0.0f;
    double t0 = now_us();
    if (which == 0) st = es_enroll(b->m, b->mix, 160000, ones, ES_ROLE_TARGET, b->e_t, v);
    else if (which == 1) st = es_cond_build(b->m, ES_MODE_PERSON, b->vt, b->vo, use_wespk, c);
    else st = es_selftest(b->m, b->est, b->est_n, &err);
    double t1 = now_us();
    if (i >= 0) t[i] = t1 - t0;
  }
  if (st != ES_OK) {
    fprintf(stderr, "%s: call returned %s\n", name, es_status_str(st));
    g_failed = 1;
  }
  emit(name, t, n, load);
  es_voiceprint_free(v);
  es_cond_free(c);
  free(ones);
  free(t);
}

/* Reads `name` (exactly `count` floats) from an open test-vector container, or NULL. */
static float *tv_floats(const es_container *ct, const char *name, int *count) {
  const es_array *a = es_container_find(ct, name);
  if (!a || a->dtype != ES_DT_F32) return NULL;
  *count = (int)(a->nbytes / sizeof(float));
  return es_tv_f32(ct, name, (size_t)*count);
}

/* Loads the model and the inputs. Returns 0, or -1 with a message on stderr. */
static int prepare(const char *dir, bench_in *b) {
  memset(b, 0, sizeof(*b));
  es_model *m = NULL;
  char path[1024], err[256];
  int have_export = dir && strcmp(dir, "synthetic") != 0 && access(dir, R_OK) == 0;
  if (have_export) {
    snprintf(path, sizeof(path), "%s/earshot.esw", dir);
    if (es_model_load_file(path, &m, err, sizeof(err)) != ES_OK) {
      fprintf(stderr, "bench: cannot load %s: %s\n", path, err);
      return -1;
    }
    static char base[256];
    snprintf(base, sizeof(base), "%s", dir);
    g_model = basename(base);                  /* "random_v1" or "v1" */
    /* testvec_long: the 10 s mixture `in`; testvec_short: enrollment audio, WeSpeaker vectors and the
       self-test file itself. */
    void *buf = NULL;
    size_t n = 0;
    es_container ct;
    snprintf(path, sizeof(path), "%s/testvec_long.est", dir);
    if (es_read_file(path, &buf, &n) == ES_OK && es_container_open(buf, n, "EST1", &ct, err, sizeof(err)) == ES_OK) {
      b->mix = tv_floats(&ct, "in", &b->n_mix);
      es_container_close(&ct);
    }
    free(buf);
    snprintf(path, sizeof(path), "%s/testvec_short.est", dir);
    if (es_read_file(path, &b->est, &b->est_n) == ES_OK &&
        es_container_open(b->est, b->est_n, "EST1", &ct, err, sizeof(err)) == ES_OK) {
      int k;
      b->enr_t = tv_floats(&ct, "enr_t", &b->Lt);
      b->enr_o = tv_floats(&ct, "enr_o", &b->Lo);
      b->e_t = tv_floats(&ct, "e_t", &k);
      b->e_o = tv_floats(&ct, "e_o", &k);
      es_container_close(&ct);
    }
    if (!b->mix || b->n_mix < 160000 || !b->enr_t || !b->enr_o) {
      fprintf(stderr, "bench: test vectors in %s are missing or short\n", dir);
      return -1;
    }
  } else {
    if (es_model_synthetic(14, 0, &m) != ES_OK) return -1;
    b->n_mix = 160000;                         /* 10 s of white noise at about -30 dBFS RMS */
    b->Lt = 96000;
    b->Lo = 64000;
    b->mix = malloc(sizeof(float) * (size_t)b->n_mix);
    b->enr_t = malloc(sizeof(float) * (size_t)b->Lt);
    b->enr_o = malloc(sizeof(float) * (size_t)b->Lo);
    for (int i = 0; i < b->n_mix; i++) b->mix[i] = 0.055f * urand();
    for (int i = 0; i < b->Lt; i++) b->enr_t[i] = 0.055f * urand();
    for (int i = 0; i < b->Lo; i++) b->enr_o[i] = 0.055f * urand();
  }
  b->m = m;
  /* Voiceprints with every frame kept (the most tokens) and the PERSON conditioning built from them. */
  int Ft = es_enroll_num_frames(b->Lt), Fo = es_enroll_num_frames(b->Lo);
  uint8_t *ones = malloc((size_t)(Ft > Fo ? Ft : Fo));
  memset(ones, 1, (size_t)(Ft > Fo ? Ft : Fo));
  b->vt = es_voiceprint_new();
  b->vo = es_voiceprint_new();
  b->c = es_cond_new();
  es_status s1 = es_enroll(m, b->enr_t, b->Lt, ones, ES_ROLE_TARGET, b->e_t, b->vt);
  es_status s2 = es_enroll(m, b->enr_o, b->Lo, ones, ES_ROLE_OWN, b->e_o, b->vo);
  es_status s3 = es_cond_build(m, ES_MODE_PERSON, b->vt, b->vo, b->e_t && b->e_o, b->c);
  free(ones);
  if (s1 != ES_OK || s2 != ES_OK || s3 != ES_OK) {
    fprintf(stderr, "bench: enrollment or conditioning failed (%s, %s, %s)\n", es_status_str(s1),
            es_status_str(s2), es_status_str(s3));
    return -1;
  }
  fprintf(stderr, "bench: model %s, conditioning with %d tokens\n", g_model, b->c->n_tokens);
  return 0;
}

int main(int argc, char **argv) {
  const char *dir = NULL;
  int ecore = 0, ecore_child = 0;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--ecore") == 0) ecore = 1;
    else if (strcmp(argv[i], "--ecore-child") == 0) ecore_child = 1;
    else if (strcmp(argv[i], "--note") == 0 && i + 1 < argc) g_note = argv[++i];
    else dir = argv[i];
  }
  /* The note goes into JSON unescaped, so refuse the characters that would break the line. */
  for (const char *q = g_note; q && *q; q++)
    if (*q == '"' || *q == '\\' || (unsigned char)*q < 0x20) {
      fprintf(stderr, "bench: --note may not contain a double quote, a backslash or a control character\n");
      return 2;
    }
  /* --ecore: start this program again under `taskpolicy -c background`, which clamps it to background
     QoS so the scheduler keeps it on the efficiency cores. The child gets --ecore-child instead. */
  if (ecore && !ecore_child) {
    char *args[16];
    int k = 0;
    args[k++] = "taskpolicy";
    args[k++] = "-c";
    args[k++] = "background";
    args[k++] = argv[0];
    args[k++] = "--ecore-child";
    if (g_note) { args[k++] = "--note"; args[k++] = (char *)g_note; }
    if (dir) args[k++] = (char *)dir;
    args[k] = NULL;
    execv("/usr/sbin/taskpolicy", args);
    perror("bench: execv taskpolicy");
    return 1;
  }
  if (ecore_child) g_suffix = "_ecore";

  mach_timebase_info_data_t tb;
  mach_timebase_info(&tb);
  g_tick_us = (double)tb.numer / (double)tb.denom / 1000.0;
  BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
  flush_denormals();

  bench_in b;
  if (prepare(dir, &b) != 0) return 1;
  row_hops(&b, 1, "hop_1inst");
  row_hops(&b, 3, "hop_3inst");
  row_render(&b, 64, 1000, 20000, "render_64");
  row_render(&b, 1104, 100, 2000, "render_1104");
  row_oneshot(&b, 0, 20, "enroll_10s");
  row_oneshot(&b, 1, 200, "cond_build");
  if (b.est) row_oneshot(&b, 2, 10, "selftest");
  else fprintf(stderr, "selftest: skipped (a synthetic model has no test vectors)\n");

  es_cond_free(b.c);
  es_voiceprint_free(b.vt);
  es_voiceprint_free(b.vo);
  free(b.mix); free(b.enr_t); free(b.enr_o); free(b.e_t); free(b.e_o); free(b.est);
  es_model_free((es_model *)b.m);
  return g_failed;
}
