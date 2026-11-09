/* esrun.c - run a WAV file through the engine offline (20-engine 14.7). A host tool, never in the app.
 *
 *   build/esrun --weights ../ml/exports/v1/earshot.esw --in mix.wav --out out.wav
 *               [--target enroll.wav] [--own own.wav] [--wespk-t e_t.f32 --wespk-o e_o.f32]
 *               [--mask-t mask_t.u8 --mask-o mask_o.u8]
 *               [--mode person|everyone] [--engine --io-rate 48000 --frames 64] [--param room_db=-24 ...]
 *
 * Stream mode (the default): the input must be 16 kHz. es_stream_hop runs hop by hop (64 samples) and
 *   the raw stream output is written at 16 kHz, same length as the input (it lags the input by 64
 *   samples, contract 2.1). No post-processing, no safety chain: exactly what parity compares.
 * Engine mode (--engine): the input must be at 48 or 44.1 kHz (it must equal --io-rate, which defaults
 *   to the file's rate). es_engine_render and es_engine_run_hops alternate on this one thread (the
 *   manual-worker hook of internal.h), with the full post and safety chain, as on the phone. The engine
 *   is unmuted at the start, so the first 400 ms fade in. The output (same length, I/O rate) lags the
 *   input by es_engine_latency_ms.
 *
 * Conditioning: --target and --own are 16 kHz enrollment WAVs (es_enroll; the mask is es_vad unless a
 *   raw uint8 mask file is given with --mask-t / --mask-o, one byte per frame, es_enroll_num_frames(n)
 *   bytes). WeSpeaker vectors are raw little-endian float32 files of 256 values written by Python;
 *   use_wespk is 1 when at least one is given, else 0. --mode defaults to person with a target and to
 *   everyone without one.
 * --param name=value sets an es_params field in engine mode (voice_db, room_db, max_gain_db,
 *   out_gain_db, slot_gain, xfade_ms, duck_own, gate, agc, bypass); es_engine_set_params clamps it.
 * Prints the model hash, the enrollment report, the latency, the stats and per-hop timing percentiles.
 * Exit status: 0 = done, 1 = bad arguments or a failed call, 2 = the output held a non-finite sample.
 * Threads: a single-threaded command-line program.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

#include "internal.h"
#include "wav.h"

/* ------------------------------------------------------------------------------------------------
 * Command-line options
 * ---------------------------------------------------------------------------------------------- */
#define MAX_PARAMS 16

typedef struct {
  const char *weights, *in, *out, *target, *own, *wespk_t, *wespk_o, *mask_t, *mask_o, *mode;
  int engine, io_rate, frames;
  const char *params[MAX_PARAMS];
  int n_params;
} options;

static void usage(void) {
  fprintf(stderr,
          "usage: esrun --weights W.esw --in mix.wav --out out.wav\n"
          "             [--target enroll.wav] [--own own.wav] [--wespk-t e_t.f32 --wespk-o e_o.f32]\n"
          "             [--mask-t mask_t.u8 --mask-o mask_o.u8]\n"
          "             [--mode person|everyone] [--engine --io-rate 48000 --frames 64] [--param name=value ...]\n");
}

/* Returns 0 when the options are complete, -1 after printing what is wrong. */
static int parse_args(int argc, char **argv, options *o) {
  memset(o, 0, sizeof(*o));
  o->frames = 64;
  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
    if (strcmp(a, "--engine") == 0) { o->engine = 1; continue; }
    if (i + 1 >= argc) { fprintf(stderr, "esrun: %s needs a value\n", a); return -1; }
    const char *v = argv[++i];
    if (strcmp(a, "--weights") == 0) o->weights = v;
    else if (strcmp(a, "--in") == 0) o->in = v;
    else if (strcmp(a, "--out") == 0) o->out = v;
    else if (strcmp(a, "--target") == 0) o->target = v;
    else if (strcmp(a, "--own") == 0) o->own = v;
    else if (strcmp(a, "--wespk-t") == 0) o->wespk_t = v;
    else if (strcmp(a, "--wespk-o") == 0) o->wespk_o = v;
    else if (strcmp(a, "--mask-t") == 0) o->mask_t = v;
    else if (strcmp(a, "--mask-o") == 0) o->mask_o = v;
    else if (strcmp(a, "--mode") == 0) o->mode = v;
    else if (strcmp(a, "--io-rate") == 0) o->io_rate = atoi(v);
    else if (strcmp(a, "--frames") == 0) o->frames = atoi(v);
    else if (strcmp(a, "--param") == 0) {
      if (o->n_params == MAX_PARAMS) { fprintf(stderr, "esrun: too many --param\n"); return -1; }
      o->params[o->n_params++] = v;
    } else {
      fprintf(stderr, "esrun: unknown option %s\n", a);
      return -1;
    }
  }
  if (!o->weights || !o->in || !o->out) { fprintf(stderr, "esrun: --weights, --in and --out are required\n"); return -1; }
  if (o->mode && strcmp(o->mode, "person") != 0 && strcmp(o->mode, "everyone") != 0) {
    fprintf(stderr, "esrun: --mode must be person or everyone\n");
    return -1;
  }
  if (o->frames < 16 || o->frames > 4096) { fprintf(stderr, "esrun: --frames must be 16..4096\n"); return -1; }
  return 0;
}

/* --param name=value into p. Returns 0, or -1 for an unknown name or a value that is not a number. */
static int apply_param(es_params *p, const char *kv) {
  const char *eq = strchr(kv, '=');
  if (!eq || eq == kv) return -1;
  size_t klen = (size_t)(eq - kv);
  char key[32];
  if (klen >= sizeof(key)) return -1;
  memcpy(key, kv, klen);
  key[klen] = 0;
  char *end = NULL;
  double v = strtod(eq + 1, &end);
  if (strcmp(key, "slot_gain") == 0) {                /* slot_gain=1,0.5,0 (up to 3 values) */
    const char *s = eq + 1;
    for (int k = 0; k < ES_MAX_INSTANCES && *s; k++) {
      p->slot_gain[k] = strtof(s, &end);
      if (end == s) return -1;
      s = *end == ',' ? end + 1 : end;
    }
    return 0;
  }
  if (end == eq + 1 || *end) return -1;
  int flag = isfinite(v) && v != 0.0;                 /* on/off fields: any nonzero number is on */
  if (strcmp(key, "voice_db") == 0) p->voice_db = (float)v;
  else if (strcmp(key, "room_db") == 0) p->room_db = (float)v;
  else if (strcmp(key, "max_gain_db") == 0) p->max_gain_db = (float)v;
  else if (strcmp(key, "out_gain_db") == 0) p->out_gain_db = (float)v;
  else if (strcmp(key, "xfade_ms") == 0) p->xfade_ms = (float)v;
  else if (strcmp(key, "duck_own") == 0) p->duck_own = flag;
  else if (strcmp(key, "gate") == 0) p->gate = flag;
  else if (strcmp(key, "agc") == 0) p->agc = flag;
  else if (strcmp(key, "bypass") == 0) p->bypass = flag;
  else return -1;
  return 0;
}

/* ------------------------------------------------------------------------------------------------
 * Small file helpers
 * ---------------------------------------------------------------------------------------------- */
/* Reads a raw file that must hold exactly `bytes` bytes into dst. 0 on success. */
static int read_exact(const char *path, void *dst, size_t bytes) {
  void *buf = NULL;
  size_t n = 0;
  if (es_read_file(path, &buf, &n) != ES_OK) { fprintf(stderr, "esrun: cannot read %s\n", path); return -1; }
  int ok = n == bytes;
  if (ok) memcpy(dst, buf, bytes);
  else fprintf(stderr, "esrun: %s holds %zu bytes, expected %zu\n", path, n, bytes);
  free(buf);
  return ok ? 0 : -1;
}

/* Reads a WAV and checks its rate (want_rate 0 = any). Returns the samples (malloc'd) or NULL. */
static float *read_wav(const char *path, int want_rate, int *n, int *rate) {
  float *pcm = NULL;
  char err[128];
  if (wav_read(path, &pcm, n, rate, err, sizeof(err)) != 0) {
    fprintf(stderr, "esrun: %s: %s\n", path, err);
    return NULL;
  }
  if (want_rate && *rate != want_rate) {
    fprintf(stderr, "esrun: %s is %d Hz, expected %d Hz (resample it first)\n", path, *rate, want_rate);
    free(pcm);
    return NULL;
  }
  return pcm;
}

/* Percentiles of n timings (microseconds), nearest-rank on a sorted copy. */
static int cmp_float(const void *a, const void *b) {
  float x = *(const float *)a, y = *(const float *)b;
  return (x > y) - (x < y);
}

static void print_timings(const char *what, float *us, size_t n) {
  if (n == 0) { printf("timing %s: no hops\n", what); return; }
  qsort(us, n, sizeof(float), cmp_float);
  const double q[5] = { 0.5, 0.9, 0.99, 0.999, 1.0 };
  float v[5];
  for (int i = 0; i < 5; i++) v[i] = us[(size_t)floor(q[i] * (double)(n - 1))];
  printf("timing %s (us over %zu hops): p50 %.1f p90 %.1f p99 %.1f p99.9 %.1f max %.1f\n", what, n, v[0], v[1],
         v[2], v[3], v[4]);
}

/* ------------------------------------------------------------------------------------------------
 * Enrollment: one 16 kHz WAV (plus an optional mask and WeSpeaker vector) into a voiceprint.
 * ---------------------------------------------------------------------------------------------- */
static es_voiceprint *enroll_one(const es_model *m, const char *wav, const char *mask_path, const char *wespk_path,
                                 es_role role) {
  const char *label = role == ES_ROLE_TARGET ? "target" : "own";
  int n = 0, rate = 0;
  float *pcm = read_wav(wav, ES_SR, &n, &rate);
  if (!pcm) return NULL;
  float e[ES_WESPK_DIM];
  uint8_t *mask = NULL;
  int F = es_enroll_num_frames(n);
  int ok = !wespk_path || read_exact(wespk_path, e, sizeof(e)) == 0;
  if (ok && mask_path) {
    mask = malloc((size_t)(F > 0 ? F : 1));
    ok = mask && read_exact(mask_path, mask, (size_t)F) == 0;
  }
  if (!ok) {                                          /* read_exact already said what is wrong */
    free(mask);
    free(pcm);
    return NULL;
  }
  es_voiceprint *v = es_voiceprint_new();
  es_status st = v ? es_enroll(m, pcm, n, mask, role, wespk_path ? e : NULL, v) : ES_ERR_NOMEM;
  if (st == ES_OK) {
    printf("enroll %s: %s, %.2f s of audio, %d frames, speech %.3f s, %d tokens, WeSpeaker %s, mask %s\n", label,
           wav, n / (double)ES_SR, F, es_voiceprint_speech_seconds(v), es_voiceprint_num_tokens(v),
           es_voiceprint_has_wespk(v) ? "yes" : "no", mask_path ? mask_path : "es_vad");
  } else {
    fprintf(stderr, "esrun: enrolling %s from %s failed: %s\n", label, wav, es_status_str(st));
    es_voiceprint_free(v);
    v = NULL;
  }
  free(mask);
  free(pcm);
  return v;
}

/* ------------------------------------------------------------------------------------------------
 * Stream mode: es_stream_hop hop by hop at 16 kHz (the tail is padded with zeros to a whole hop and the
 * output cut back to n samples).
 * ---------------------------------------------------------------------------------------------- */
static int run_stream(const es_model *m, const es_cond *c, const float *in, int n, float *out) {
  es_stream *s = es_stream_new(m);
  int hops = (n + ES_HOP - 1) / ES_HOP;
  float *times = malloc(sizeof(float) * (size_t)(hops > 0 ? hops : 1));
  if (!s || !times) {
    fprintf(stderr, "esrun: out of memory\n");
    es_stream_free(s);
    free(times);
    return 1;
  }
  mach_timebase_info_data_t tb;
  mach_timebase_info(&tb);
  float x[ES_HOP], y[ES_HOP], hd[2];
  double pt_sum = 0.0, ps_sum = 0.0;
  int nan_resets = 0;
  for (int h = 0; h < hops; h++) {
    int base = h * ES_HOP, len = n - base < ES_HOP ? n - base : ES_HOP;
    memset(x, 0, sizeof(x));
    memcpy(x, in + base, sizeof(float) * (size_t)len);
    uint64_t t0 = mach_absolute_time();
    nan_resets += es_stream_hop(s, c, x, y, hd);
    uint64_t dt = mach_absolute_time() - t0;
    times[h] = (float)((double)dt * tb.numer / tb.denom / 1000.0);
    memcpy(out + base, y, sizeof(float) * (size_t)len);
    pt_sum += hd[0];
    ps_sum += hd[1];
  }
  printf("latency: stream lag 64 samples = 4.000 ms at 16 kHz (algorithmic latency from capture 128 samples = "
         "8.000 ms, contract 2.1)\n");
  printf("stats: hops %d, nan_resets %d, mean p_target %.4f, mean p_self %.4f\n", hops, nan_resets,
         hops ? pt_sum / hops : 0.0, hops ? ps_sum / hops : 0.0);
  print_timings("es_stream_hop", times, (size_t)hops);
  free(times);
  es_stream_free(s);
  return 0;
}

/* ------------------------------------------------------------------------------------------------
 * Engine mode: render + run_hops alternate on this thread, callbacks of --frames samples (the last one
 * may be shorter), one instance in slot 0, AirPods default slack 43.
 * ---------------------------------------------------------------------------------------------- */
static int run_engine(const es_model *m, const es_cond *c, const options *o, const float *in, int n, int rate,
                      float *out) {
  es_engine_config cfg = { rate, o->frames, 4096, 1, 43 };
  es_engine *e = NULL;
  es_status st = es_engine_new(m, &cfg, &e);
  if (st != ES_OK) {
    fprintf(stderr, "esrun: es_engine_new(%d Hz, %d frames) failed: %s\n", rate, o->frames, es_status_str(st));
    return 1;
  }
  st = es_engine_set_cond(e, 0, c, 1);              /* no worker yet, so it is picked up at once */
  if (st != ES_OK) {
    fprintf(stderr, "esrun: es_engine_set_cond failed: %s\n", es_status_str(st));
    es_engine_free(e);
    return 1;
  }
  es_engine_set_manual_worker(e, 1);                /* this thread runs the hops itself */
  es_params p;
  es_params_default(&p);
  for (int i = 0; i < o->n_params; i++)
    if (apply_param(&p, o->params[i]) != 0) {
      fprintf(stderr, "esrun: bad --param %s\n", o->params[i]);
      es_engine_free(e);
      return 1;
    }
  es_engine_set_params(e, &p);                      /* clamps every field (contract ranges) */
  es_engine_set_mute(e, 0);                         /* a new engine starts muted: 400 ms fade-in */

  size_t cap = (size_t)((double)n * ES_SR / rate / ES_HOP) + 2048, n_t = 0;
  float *times = malloc(sizeof(float) * cap);
  if (!times) {
    fprintf(stderr, "esrun: out of memory\n");
    es_engine_free(e);
    return 1;
  }
  uint16_t tbuf[ES_TIMINGS_SIZE];
  int max_hops = 0;
  for (int off = 0; off < n; off += o->frames) {
    int k = n - off < o->frames ? n - off : o->frames;
    es_engine_render(e, in + off, out + off, k);
    int h = es_engine_run_hops(e);
    if (h > max_hops) max_hops = h;
    int got = es_engine_read_timings(e, tbuf, ES_TIMINGS_SIZE);   /* drain the 1,024-entry ring */
    for (int i = 0; i < got && n_t < cap; i++) times[n_t++] = (float)tbuf[i];
  }
  es_stats s;
  es_engine_get_stats(e, &s);
  printf("latency: es_engine_latency_ms %.4f ms (P_min %d + slack16 %d at 16 kHz, plus both resamplers)\n",
         es_engine_latency_ms(e), e->p_min, s.slack16);
  printf("stats: hops %llu, underruns %llu, nan_resets %llu, slack16 %d, howl %d, agc_gain_db %.2f, "
         "limiter_gr_db %.2f, in_rms_db %.1f, out_rms_db %.1f, compute_us_max %.0f, most hops per run_hops %d\n",
         (unsigned long long)s.hops, (unsigned long long)s.underruns, (unsigned long long)s.nan_resets, s.slack16,
         s.howl, s.agc_gain_db + 0.0f, s.limiter_gr_db + 0.0f, s.in_rms_db, s.out_rms_db, s.compute_us_max, max_hops);
  print_timings("engine hop (worker body)", times, n_t);   /* "+ 0.0f" above prints -0 as 0 */
  free(times);
  es_engine_free(e);
  return 0;
}

/* ------------------------------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------------------------------- */
int main(int argc, char **argv) {
  options o;
  if (parse_args(argc, argv, &o) != 0) {
    usage();
    return 1;
  }
  char err[256];
  es_model *m = NULL;
  es_status st = es_model_load_file(o.weights, &m, err, sizeof(err));
  if (st != ES_OK) {
    fprintf(stderr, "esrun: loading %s failed: %s (%s)\n", o.weights, es_status_str(st), err);
    return 1;
  }
  printf("model: %s\n  model_hash  %s\n  config_hash %s\n", o.weights, es_model_data_hash(m), es_model_config_hash(m));

  /* Conditioning. */
  int person = o.mode ? strcmp(o.mode, "person") == 0 : o.target != NULL;
  int rc = 1;
  es_voiceprint *vt = NULL, *vo = NULL;
  es_cond *c = NULL;
  float *in = NULL, *out = NULL;
  if (person && !o.target) { fprintf(stderr, "esrun: person mode needs --target\n"); goto done; }
  if (!person && o.target) printf("note: --target is not used in everyone mode\n");
  if (person && !(vt = enroll_one(m, o.target, o.mask_t, o.wespk_t, ES_ROLE_TARGET))) goto done;
  if (o.own && !(vo = enroll_one(m, o.own, o.mask_o, o.wespk_o, ES_ROLE_OWN))) goto done;
  int use_wespk = o.wespk_t != NULL || o.wespk_o != NULL;
  c = es_cond_new();
  st = c ? es_cond_build(m, person ? ES_MODE_PERSON : ES_MODE_EVERYONE, vt, vo, use_wespk, c) : ES_ERR_NOMEM;
  if (st != ES_OK) { fprintf(stderr, "esrun: es_cond_build failed: %s\n", es_status_str(st)); goto done; }
  printf("conditioning: mode %s, own voice %s, use_wespk %d, %d tokens\n", person ? "person" : "everyone",
         vo ? "yes" : "no", use_wespk, c->n_tokens);

  /* Input, run, output. */
  int n = 0, rate = 0;
  in = read_wav(o.in, o.engine ? o.io_rate : ES_SR, &n, &rate);
  if (!in) goto done;
  out = calloc((size_t)(n > 0 ? n : 1), sizeof(float));
  if (!out) { fprintf(stderr, "esrun: out of memory\n"); goto done; }
  printf("input: %s, %d samples at %d Hz (%.2f s), %s mode\n", o.in, n, rate, n / (double)rate,
         o.engine ? "engine" : "stream");
  if (!o.engine && o.n_params) printf("note: --param is ignored in stream mode (no post-processing)\n");
  rc = o.engine ? run_engine(m, c, &o, in, n, rate, out) : run_stream(m, c, in, n, out);
  if (rc != 0) goto done;
  int bad = 0;
  float peak = 0.0f;
  for (int i = 0; i < n; i++) {
    if (!isfinite(out[i])) bad++;
    else if (fabsf(out[i]) > peak) peak = fabsf(out[i]);
  }
  if (wav_write(o.out, out, n, rate, 0, err, sizeof(err)) != 0) {
    fprintf(stderr, "esrun: writing %s failed: %s\n", o.out, err);
    rc = 1;
    goto done;
  }
  printf("output: %s, %d samples at %d Hz (float32), peak %.6f, non-finite samples %d\n", o.out, n, rate, peak, bad);
  rc = bad ? 2 : 0;
done:
  free(in);
  free(out);
  es_cond_free(c);
  es_voiceprint_free(vt);
  es_voiceprint_free(vo);
  es_model_free(m);
  return rc;
}
