/* test_parity.c - the C engine against the PyTorch export (20-engine 14.2, contract 3.3).
 *
 * Usage: build/test_parity [EXPORT_DIR]   (default ../ml/exports/random_v1; make parity runs it with
 *        VECLIB_MAXIMUM_THREADS=1 so BLAS is single-threaded and the determinism check is meaningful)
 *
 * Loads EXPORT_DIR/earshot.esw, then for testvec_short.est and testvec_long.est:
 *   1. meta.tolerances must hold exactly the 10 keys of contract 3.3; meta.weights_sha256 must equal
 *      es_model_data_hash
 *   2. es_vad, and the token encoder (tokens, g) for the target and the own voice
 *   3. es_cond_build for the 4 cases (person, everyone, noown, nowespk) vs cond.{case}.*
 *   4. per case the stream runs (a) with the file's reference conditioning, FZ off and FZ on, and
 *      (b) with the engine-built conditioning (end to end); out, out SNR and heads checked each time
 *   5. short file only: the debug taps of the first 8 hops of `person` (dbg.feat, z_enc, z_blk0..3,
 *      coef); dbg.feat uses the DECISIONS rule for near-zero bins (|X| < 1e-4 are reported, not failed)
 *   6. bit determinism: every case run twice from reset, memcmp; short file only, the whole person path
 *      (enroll, cond, stream) run again with the process held on the efficiency cores, memcmp
 *   7. max abs error and SNR printed for every array; a summary per tolerance key with its margin
 * Also (short file): target token selection above 48 groups (the vectors stop at 45), and a header whose
 * meta.tolerances loosens a contract value is refused by es_tv_tolerances and es_selftest.
 * Also checks es_model_load from memory and from a file, loader errors that name the tensor, and
 * es_selftest. Every check prints "PASS name value limit" or "FAIL ..."; exit 1 on any failure.
 */
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#include "internal.h"

/* ---------------------------------------------------------------------------------------------- */
/* Harness                                                                                         */
/* ---------------------------------------------------------------------------------------------- */
static int g_pass = 0, g_fail = 0;

static void check(const char *name, int ok, double value, double limit) {
  printf("%s %s %.6g %.6g\n", ok ? "PASS" : "FAIL", name, value, limit);
  if (ok) g_pass++; else g_fail++;
}
static void check_le(const char *name, double value, double limit) { check(name, value <= limit, value, limit); }
static void check_ge(const char *name, double value, double limit) { check(name, value >= limit, value, limit); }
static void check_true(const char *name, int ok) { check(name, ok, ok ? 1 : 0, 1); }

/* Worst error seen per tolerance key, per file (0 = short, 1 = long), for the final summary. */
enum { K_VAD, K_TOKENS, K_G, K_FILM, K_K, K_V, K_DBG, K_OUT, K_SNR, K_HEADS, K_COUNT };
static const char *KEY_NAMES[K_COUNT] = { "vad", "tokens", "g", "film", "k", "v", "dbg", "out", "out_snr_db", "heads" };
static double g_worst[2][K_COUNT];      /* errors: max; out_snr_db: min */
static double g_limit[2][K_COUNT];
static int g_seen[2][K_COUNT];

static void note(int file, int key, double value) {
  if (!g_seen[file][key]) { g_worst[file][key] = value; g_seen[file][key] = 1; return; }
  if (key == K_SNR) g_worst[file][key] = fmin(g_worst[file][key], value);
  else              g_worst[file][key] = fmax(g_worst[file][key], value);
}

/* Flush-to-zero (FPCR.FZ, bit 24) on or off for this thread (20-engine 3.6). */
static void set_fz(int on) {
#if defined(__aarch64__)
  uint64_t r;
  __asm__ volatile("mrs %0, fpcr" : "=r"(r));
  if (on) r |= 1ull << 24; else r &= ~(1ull << 24);
  __asm__ volatile("msr fpcr, %0" : : "r"(r));
#else
  (void)on;
#endif
}

/* 1 if FPCR.FZ is set on this thread (used to prove that set_fz took effect). */
static int get_fz(void) {
#if defined(__aarch64__)
  uint64_t r;
  __asm__ volatile("mrs %0, fpcr" : "=r"(r));
  return (int)((r >> 24) & 1);
#else
  return -1;
#endif
}

static char *join(const char *dir, const char *file) {
  size_t n = strlen(dir) + strlen(file) + 2;
  char *p = malloc(n);
  snprintf(p, n, "%s/%s", dir, file);
  return p;
}

/* ---------------------------------------------------------------------------------------------- */
/* Loader checks (20-engine 4.3)                                                                   */
/* ---------------------------------------------------------------------------------------------- */
/* Replace the first `from` inside the JSON header of a copy of the file with `to` (same length, so
   every offset and the data hash stay valid) and try to load it. Returns the status; err gets the
   loader's message. */
/* A malloc'd copy of a container file with the first `from` inside its JSON header replaced by `to`
   (same length), or NULL if `from` is not in the header. */
static uint8_t *patch_copy(const uint8_t *buf, size_t n, const char *from, const char *to) {
  uint32_t header_len;
  memcpy(&header_len, buf + 8, 4);
  size_t fl = strlen(from);
  if (strlen(to) != fl) return NULL;
  for (size_t i = 16; i + fl <= 16 + (size_t)header_len && i + fl <= n; i++) {
    if (memcmp(buf + i, from, fl) != 0) continue;
    uint8_t *copy = malloc(n);
    memcpy(copy, buf, n);
    memcpy(copy + i, to, fl);
    return copy;
  }
  return NULL;
}

static es_status load_patched(const uint8_t *buf, size_t n, const char *from, const char *to, char *err, size_t el) {
  uint8_t *copy = patch_copy(buf, n, from, to);
  es_model *m = NULL;
  es_status st = copy ? es_model_load(copy, n, &m, err, el) : ES_ERR_ARG;
  es_model_free(m);
  free(copy);
  return st;
}

static es_model *test_loader(const char *dir) {
  char err[256];
  char *path = join(dir, "earshot.esw");
  void *buf = NULL;
  size_t n = 0;
  if (es_read_file(path, &buf, &n) != ES_OK) {
    printf("FAIL read_esw cannot read %s\n", path);
    g_fail++;
    free(path);
    return NULL;
  }

  /* From memory; then the file buffer is freed and the model must still work (it copied the data). */
  es_model *m = NULL;
  es_status st = es_model_load(buf, n, &m, err, sizeof(err));
  check_true("load_esw_from_memory", st == ES_OK && m != NULL);
  if (st != ES_OK) printf("  loader said: %s\n", err);
  es_model *mf = NULL;
  st = es_model_load_file(path, &mf, err, sizeof(err));
  check_true("load_esw_from_file", st == ES_OK && mf != NULL);
  if (m && mf) {
    check_true("load_file_and_memory_identical", m->arena_bytes == mf->arena_bytes &&
                                                 memcmp(m->arena, mf->arena, m->arena_bytes) == 0);
    check_true("model_hashes", strcmp(es_model_config_hash(m), ES_CONFIG_HASH_HEX) == 0 &&
                               strlen(es_model_data_hash(m)) == 64 &&
                               strcmp(es_model_data_hash(m), es_model_data_hash(mf)) == 0);
    check_true("model_latency_128", es_model_latency_samples(m) == 128);
    /* Every bound pointer is 64-byte aligned inside the arena. */
    static es_tensor list[ES_N_TENSORS];
    int nt = es_model_tensors(m, list), aligned = 1;    /* list[i].dst = address of the field of tensor i */
    (void)nt;
    for (int i = 0; i < ES_N_TENSORS; i++) {
      const char *p = (const char *)*list[i].dst, *base = m->arena;
      if (!p || p < base || p >= base + m->arena_bytes || (size_t)(p - base) % 64 != 0) aligned = 0;
    }
    check_true("load_270_tensors_bound_aligned", aligned);
    printf("INFO model_hash %s\n", es_model_data_hash(m));
  }
  es_model_free(mf);

  /* Loader errors name the tensor (the data hash still matches because only the header changes). */
  st = load_patched(buf, n, "\"blocks.2.t.w_hh\"", "\"blocks.2.t.w_hX\"", err, sizeof(err));
  check_true("load_missing_tensor_named", st == ES_ERR_FORMAT && strstr(err, "blocks.2.t.w_hh") != NULL);
  printf("  loader said: %s\n", err);
  st = load_patched(buf, n, "\"shape\":[24,64]", "\"shape\":[64,24]", err, sizeof(err));
  check_true("load_wrong_shape_named", st == ES_ERR_FORMAT && strstr(err, "enc.lin.0.weight") != NULL &&
                                       strstr(err, "[64,24]") != NULL);
  printf("  loader said: %s\n", err);
  es_model *bad = NULL;
  st = es_model_load(buf, n - 1, &bad, err, sizeof(err));
  check_true("load_truncated_rejected", st == ES_ERR_FORMAT && bad == NULL);
  check_true("load_null_args", es_model_load(NULL, n, &bad, err, sizeof(err)) == ES_ERR_ARG &&
                               es_model_load(buf, n, NULL, err, sizeof(err)) == ES_ERR_ARG);
  free(buf);
  free(path);
  return m;
}

/* ---------------------------------------------------------------------------------------------- */
/* One test-vector file                                                                            */
/* ---------------------------------------------------------------------------------------------- */
typedef struct { const char *name; es_mode mode; int has_t, has_o, use_wespk; } tcase;
/* Contract 3.3 and 20-engine 7.4: person, everyone, noown, nowespk. */
static const tcase CASES[4] = {
  { "person", ES_MODE_PERSON, 1, 1, 1 },
  { "everyone", ES_MODE_EVERYONE, 0, 1, 1 },
  { "noown", ES_MODE_PERSON, 1, 0, 1 },
  { "nowespk", ES_MODE_PERSON, 1, 1, 0 },
};

/* f32 array that must exist with `count` floats; prints a FAIL line when it does not. */
static float *need_f32(const es_container *ct, const char *name, size_t count) {
  float *x = es_tv_f32(ct, name, count);
  if (!x) { printf("FAIL array_%s missing or wrong size\n", name); g_fail++; }
  return x;
}

static int array_len(const es_container *ct, const char *name, int dim) {
  const es_array *a = es_container_find(ct, name);
  return a && a->ndim > dim ? a->shape[dim] : -1;
}

/* The file's reference conditioning set for case `cs`, in an engine es_cond. Returns 0 or -1. */
static int load_ref_cond(const es_container *ct, const char *cs, es_mode mode, es_cond *c) {
  char name[64];
  float *film, *k, *v;
  snprintf(name, sizeof(name), "cond.%s.film", cs); film = need_f32(ct, name, 4096);
  snprintf(name, sizeof(name), "cond.%s.k", cs);    k = need_f32(ct, name, 4 * 68 * 16);
  snprintf(name, sizeof(name), "cond.%s.v", cs);    v = need_f32(ct, name, 4 * 68 * 16);
  snprintf(name, sizeof(name), "cond.%s.n", cs);
  const es_array *na = es_container_find(ct, name);
  int ok = film && k && v && na && na->dtype == ES_DT_I32 && na->nbytes == 4;
  if (ok) {
    memcpy(c->film, film, sizeof(c->film));
    memcpy(c->k, k, sizeof(c->k));
    memcpy(c->v, v, sizeof(c->v));
    memcpy(&c->n_tokens, na->data, 4);
    c->mode = (int32_t)mode;
  }
  free(film); free(k); free(v);
  return ok ? 0 : -1;
}

/* Run a fresh stream over `in` (N samples) with conditioning c. Returns the number of NaN resets. */
static int run_stream(const es_model *m, const es_cond *c, const float *in, int N, float *out, float *heads) {
  es_stream *s = es_stream_new(m);
  int resets = 0;
  for (int j = 0; j < N / ES_HOP; j++)
    resets += es_stream_hop(s, c, in + (size_t)j * ES_HOP, out + (size_t)j * ES_HOP, heads + (size_t)j * 2);
  es_stream_free(s);
  return resets;
}

/* 1 if one of the strings in meta.notes contains `text` (the exporter writes there how each mask was made). */
static int meta_note_has(const es_container *ct, const char *text) {
  const es_json *j = &ct->j;
  int arr = es_json_get_path(j, 0, "meta.notes");
  if (arr < 0 || j->tok[arr].type != ES_JSON_ARRAY) return 0;
  int tl = (int)strlen(text);
  for (int i = 0; i < j->tok[arr].size; i++) {
    int t = es_json_arr_at(j, arr, i);
    if (t < 0 || j->tok[t].type != ES_JSON_STRING) continue;
    for (int p = j->tok[t].start; p + tl <= j->tok[t].end; p++)   /* start/end bound the string's bytes */
      if (memcmp(j->js + p, text, (size_t)tl) == 0) return 1;
  }
  return 0;
}

/* The whole person path: enroll the target and the own voice with the file's masks, build
   PERSON(target, own, 1), stream all N samples from reset. The cross-core determinism check runs it once
   as usual and once with the process held on the efficiency cores. */
typedef struct {
  const es_model *m;
  const float *enr_t, *enr_o, *e_t, *e_o, *in;
  const uint8_t *mask_t, *mask_o;
  int Lt, Lo, N;
  es_voiceprint *vt, *vo;   /* results (the caller frees them) */
  es_cond *cond;
  float *out, *heads;
  int ok;
  double ms;                /* wall time, to show which kind of core ran it */
} e2e_job;

static void run_e2e(e2e_job *j) {
  mach_timebase_info_data_t tb;
  mach_timebase_info(&tb);
  uint64_t t0 = mach_absolute_time();
  j->vt = es_voiceprint_new();
  j->vo = es_voiceprint_new();
  j->cond = es_cond_new();
  j->out = malloc(sizeof(float) * (size_t)j->N);
  j->heads = malloc(sizeof(float) * (size_t)(j->N / ES_HOP) * 2);
  j->ok = j->vt && j->vo && j->cond && j->out && j->heads &&
          es_enroll(j->m, j->enr_t, j->Lt, j->mask_t, ES_ROLE_TARGET, j->e_t, j->vt) == ES_OK &&
          es_enroll(j->m, j->enr_o, j->Lo, j->mask_o, ES_ROLE_OWN, j->e_o, j->vo) == ES_OK &&
          es_cond_build(j->m, ES_MODE_PERSON, j->vt, j->vo, 1, j->cond) == ES_OK;
  if (j->ok) run_stream(j->m, j->cond, j->in, j->N, j->out, j->heads);
  j->ms = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;
}

static void e2e_free(e2e_job *j) {
  es_voiceprint_free(j->vt);
  es_voiceprint_free(j->vo);
  es_cond_free(j->cond);
  free(j->out);
  free(j->heads);
}

/* 1 if the two runs are bit-identical in tokens, g, conditioning, output and heads. */
static int e2e_same(const e2e_job *a, const e2e_job *b) {
  return a->ok && b->ok && memcmp(a->vt, b->vt, sizeof(es_voiceprint)) == 0 &&
         memcmp(a->vo, b->vo, sizeof(es_voiceprint)) == 0 && memcmp(a->cond, b->cond, sizeof(es_cond)) == 0 &&
         memcmp(a->out, b->out, sizeof(float) * (size_t)a->N) == 0 &&
         memcmp(a->heads, b->heads, sizeof(float) * (size_t)(a->N / ES_HOP) * 2) == 0;
}

/* Check one streamed run against out.{case} and heads.{case}. */
static void check_run(int file, const char *tag, const float *out, const float *heads, const float *out_ref,
                      const float *heads_ref, int N, int resets, const es_tv_tol *tol) {
  char name[96];
  double e_out = es_max_abs_diff(out, out_ref, (size_t)N), snr = es_snr_db(out, out_ref, (size_t)N);
  double e_heads = es_max_abs_diff(heads, heads_ref, (size_t)(N / ES_HOP) * 2);
  snprintf(name, sizeof(name), "%s_out", tag);        check_le(name, e_out, tol->out);
  snprintf(name, sizeof(name), "%s_out_snr_db", tag); check_ge(name, snr, tol->out_snr_db);
  snprintf(name, sizeof(name), "%s_heads", tag);      check_le(name, e_heads, tol->heads);
  snprintf(name, sizeof(name), "%s_no_nan_reset", tag); check_true(name, resets == 0);
  note(file, K_OUT, e_out);
  note(file, K_SNR, snr);
  note(file, K_HEADS, e_heads);
}

/* Debug taps of the first 8 hops of `person` (short file only, 20-engine 14.2 step 5). */
static void check_dbg(const es_model *m, const es_container *ct, const es_cond *ref, const float *in,
                      const es_tv_tol *tol) {
  float *feat = need_f32(ct, "dbg.feat", 8 * 771), *z_enc = need_f32(ct, "dbg.z_enc", 8 * 1024);
  float *coef = need_f32(ct, "dbg.coef", 8 * 1026), *z_blk[4];
  char name[64];
  for (int b = 0; b < 4; b++) {
    snprintf(name, sizeof(name), "dbg.z_blk%d", b);
    z_blk[b] = need_f32(ct, name, 8 * 1024);
  }
  if (!feat || !z_enc || !coef || !z_blk[0] || !z_blk[1] || !z_blk[2] || !z_blk[3]) goto done;

  /* Engine taps for the same 8 hops, from reset, with the reference conditioning. */
  float e_feat[8][771], e_zenc[8][1024], e_zblk[4][8][1024], e_coef[8][1026], out[64], h[2];
  es_stream *s = es_stream_new(m);
  for (int j = 0; j < 8; j++) {
    es_dbg d = { e_feat[j], e_zenc[j], { e_zblk[0][j], e_zblk[1][j], e_zblk[2][j], e_zblk[3][j] }, e_coef[j] };
    es_stream_hop_dbg(s, ref, in + 64 * j, out, h, &d);
  }
  es_stream_free(s);

  /* dbg.feat, with the DECISIONS rule: bins whose reference |X| is below 1e-4 are a float32 FFT
     rounding artifact (vDSP vs torch), so they are reported but not failed. |X| comes from the
     reference log-power feature: feat[3 bin + 2] = 0.1 ln(|X|^2 + 1e-8). */
  double e_big = 0, e_small = 0, e_log_all = 0;
  int n_small = 0;
  for (int j = 0; j < 8; j++)
    for (int bin = 0; bin < 257; bin++) {
      const float *r = feat + j * 771 + 3 * bin, *e = e_feat[j] + 3 * bin;
      double mag2 = exp(10.0 * (double)r[2]) - 1e-8;
      double err_c = fmax(fabs((double)e[0] - r[0]), fabs((double)e[1] - r[1]));
      e_log_all = fmax(e_log_all, fabs((double)e[2] - r[2]));
      if (mag2 >= 1e-8) e_big = fmax(e_big, err_c);   /* |X| >= 1e-4 */
      else { n_small++; e_small = fmax(e_small, err_c); }
    }
  double e_feat_checked = fmax(e_big, e_log_all);
  check_le("dbg_feat_bins_with_absX_ge_1e-4", e_feat_checked, tol->dbg);
  printf("INFO dbg_feat_near_zero_bins count %d of %d, max abs error there %.3g (not failed, DECISIONS)\n",
         n_small, 8 * 257, e_small);
  printf("INFO dbg_feat_all_bins_max_abs_error %.3g (limit %.3g)\n", fmax(e_feat_checked, e_small), tol->dbg);
  note(0, K_DBG, e_feat_checked);

  double e = es_max_abs_diff(&e_zenc[0][0], z_enc, 8 * 1024);
  check_le("dbg_z_enc", e, tol->dbg); note(0, K_DBG, e);
  for (int b = 0; b < 4; b++) {
    e = es_max_abs_diff(&e_zblk[b][0][0], z_blk[b], 8 * 1024);
    snprintf(name, sizeof(name), "dbg_z_blk%d", b);
    check_le(name, e, tol->dbg); note(0, K_DBG, e);
  }
  e = es_max_abs_diff(&e_coef[0][0], coef, 8 * 1026);
  check_le("dbg_coef", e, tol->dbg); note(0, K_DBG, e);
done:
  free(feat); free(z_enc); free(coef);
  for (int b = 0; b < 4; b++) free(z_blk[b]);
}

/* All checks of 20-engine 14.2 for one .est file. file = 0 (short) or 1 (long). */
static void test_file(const es_model *m, const char *dir, const char *fname, int file) {
  char err[256], name[96];
  char *path = join(dir, fname);
  void *buf = NULL;
  size_t nbytes = 0;
  printf("== %s\n", path);
  es_container ct;
  if (es_read_file(path, &buf, &nbytes) != ES_OK || es_container_open(buf, nbytes, "EST1", &ct, err, sizeof(err))) {
    printf("FAIL open_%s %s\n", fname, err);
    g_fail++;
    free(buf); free(path);
    return;
  }

  /* 1. Tolerances (exact key set) and the weights hash. */
  es_tv_tol tol;
  es_status st = es_tv_tolerances(&ct, &tol, err, sizeof(err));
  check_true("tolerances_exact_keys_and_contract_values", st == ES_OK);
  if (st != ES_OK) { printf("  %s\n", err); es_container_close(&ct); free(buf); free(path); return; }
  double lim[K_COUNT] = { tol.vad, tol.tokens, tol.g, tol.film, tol.k, tol.v, tol.dbg, tol.out, tol.out_snr_db, tol.heads };
  memcpy(g_limit[file], lim, sizeof(lim));
  check_true("weights_sha256_matches_model",
             es_json_str_eq(&ct.j, es_json_get_path(&ct.j, 0, "meta.weights_sha256"), es_model_data_hash(m)));

  /* Inputs. */
  int Lt = array_len(&ct, "enr_t", 0), Lo = array_len(&ct, "enr_o", 0), N = array_len(&ct, "in", 0);
  int nt = array_len(&ct, "tok_t", 0), no = array_len(&ct, "tok_o", 0);
  int Ft = es_enroll_num_frames(Lt), Fo = es_enroll_num_frames(Lo);
  float *enr_t = need_f32(&ct, "enr_t", (size_t)Lt), *enr_o = need_f32(&ct, "enr_o", (size_t)Lo);
  float *e_t = need_f32(&ct, "e_t", 256), *e_o = need_f32(&ct, "e_o", 256), *in = need_f32(&ct, "in", (size_t)N);
  float *tok_t = need_f32(&ct, "tok_t", (size_t)nt * 64), *tok_o = need_f32(&ct, "tok_o", (size_t)no * 64);
  float *g_t = need_f32(&ct, "g_t", 128), *g_o = need_f32(&ct, "g_o", 128);
  const es_array *a_mt = es_container_find(&ct, "mask_t"), *a_mo = es_container_find(&ct, "mask_o");
  const es_array *a_vt = es_container_find(&ct, "vad_t");
  if (!enr_t || !enr_o || !e_t || !e_o || !in || !tok_t || !tok_o || !g_t || !g_o || !a_mt || !a_mo || !a_vt ||
      (int)a_mt->nbytes != Ft || (int)a_mo->nbytes != Fo || (int)a_vt->nbytes != Ft || N % 64 != 0) {
    printf("FAIL inputs_present\n");
    g_fail++;
    goto out;
  }
  const uint8_t *mask_t = a_mt->data, *mask_o = a_mo->data, *vad_t = a_vt->data;
  printf("INFO N %d, enrollment %d / %d samples, F %d / %d, tokens %d / %d\n", N, Lt, Lo, Ft, Fo, nt, no);

  /* 2a. VAD: es_vad(enr_t) == vad_t (contract 3.3). es_vad(enr_o) is compared with mask_o only when the
     file says mask_o was made that way (meta.notes "mask_o = es_vad(enr_o)"); the contract does not
     promise it, and an export with an oracle mask_o must not fail here. */
  uint8_t *vad = malloc((size_t)(Ft > Fo ? Ft : Fo));
  int mism_t = 0, mism_o = 0;
  es_vad(enr_t, Lt, vad);
  for (int f = 0; f < Ft; f++) mism_t += vad[f] != vad_t[f];
  es_vad(enr_o, Lo, vad);
  for (int f = 0; f < Fo; f++) mism_o += vad[f] != mask_o[f];
  free(vad);
  check_le("vad_target_mismatches", mism_t, tol.vad);
  int own_from_vad = meta_note_has(&ct, "mask_o = es_vad(enr_o)");
  if (own_from_vad) check_le("vad_own_vs_mask_o_mismatches", mism_o, tol.vad);
  else printf("INFO vad_own_vs_mask_o not checked (meta.notes does not say mask_o = es_vad(enr_o)), %d mismatches\n",
              mism_o);
  note(file, K_VAD, own_from_vad ? fmax(mism_t, mism_o) : mism_t);

  /* 2b. Token encoder with the file's masks (mask_t has frames 40..49 forced to 0). */
  es_voiceprint *vt = es_voiceprint_new(), *vo = es_voiceprint_new();
  int st_t = es_enroll(m, enr_t, Lt, mask_t, ES_ROLE_TARGET, e_t, vt);
  int st_o = es_enroll(m, enr_o, Lo, mask_o, ES_ROLE_OWN, e_o, vo);
  check_true("enroll_target_ok_token_count", st_t == ES_OK && vt->n_tok == nt);
  check_true("enroll_own_ok_token_count", st_o == ES_OK && vo->n_tok == no);
  double e_tok_t = es_max_abs_diff(&vt->tokens[0][0], tok_t, (size_t)nt * 64);
  double e_tok_o = es_max_abs_diff(&vo->tokens[0][0], tok_o, (size_t)no * 64);
  double e_g_t = es_max_abs_diff(vt->g, g_t, 128), e_g_o = es_max_abs_diff(vo->g, g_o, 128);
  check_le("tokens_target", e_tok_t, tol.tokens);
  check_le("tokens_own", e_tok_o, tol.tokens);
  check_le("g_target", e_g_t, tol.g);
  check_le("g_own", e_g_o, tol.g);
  note(file, K_TOKENS, fmax(e_tok_t, e_tok_o));
  note(file, K_G, fmax(e_g_t, e_g_o));
  printf("INFO speech_s target %.3f own %.3f, has_wespk %d %d\n", es_voiceprint_speech_seconds(vt),
         es_voiceprint_speech_seconds(vo), es_voiceprint_has_wespk(vt), es_voiceprint_has_wespk(vo));

  /* 2c. Target token selection (contract 2.4 step 5), which the vectors never reach (n_t = 45 < 48).
     An all-ones mask on enr_t gives n_avail = Ft / 3 groups (62 for Ft = 188), so 48 are picked with
     floor(i * n_avail / 48). Enrolling instead with a mask that holds only the frames of those 48 groups
     (so no selection happens) must give bit-identical tokens and g, because the mask only steers the
     pooling, never the network. The PyTorch side of this rule is not covered by the vectors (WP-ML). */
  if (file == 0 && Ft / 3 > ES_MAX_TGT_TOKENS) {
    int n_av = Ft / 3;
    uint8_t *ones = malloc((size_t)Ft), *picked = calloc((size_t)Ft, 1);
    memset(ones, 1, (size_t)Ft);
    for (int i = 0; i < ES_MAX_TGT_TOKENS; i++) {
      int grp = (int)((int64_t)i * n_av / ES_MAX_TGT_TOKENS);   /* group grp = frames 3 grp .. 3 grp + 2 */
      for (int r = 0; r < 3; r++) picked[3 * grp + r] = 1;
    }
    es_voiceprint *va = es_voiceprint_new(), *vb = es_voiceprint_new();
    int sa = es_enroll(m, enr_t, Lt, ones, ES_ROLE_TARGET, e_t, va);
    int sb = es_enroll(m, enr_t, Lt, picked, ES_ROLE_TARGET, e_t, vb);
    snprintf(name, sizeof(name), "tokens_target_selection_%d_to_48_matches_formula", n_av);
    check(name, sa == ES_OK && sb == ES_OK && va->n_tok == 48 && vb->n_tok == 48 &&
                memcmp(va->tokens, vb->tokens, sizeof(va->tokens)) == 0 && memcmp(va->g, vb->g, sizeof(va->g)) == 0,
          n_av, ES_MAX_TGT_TOKENS);
    es_voiceprint_free(va);
    es_voiceprint_free(vb);
    free(ones);
    free(picked);
  }

  /* 3. Conditioning for the 4 cases, built by the engine, vs cond.{case}.*. */
  es_cond *eng[4], *ref[4];
  for (int i = 0; i < 4; i++) {
    const tcase *tc = &CASES[i];
    eng[i] = es_cond_new();
    ref[i] = es_cond_new();
    st = es_cond_build(m, tc->mode, tc->has_t ? vt : NULL, tc->has_o ? vo : NULL, tc->use_wespk, eng[i]);
    int have_ref = load_ref_cond(&ct, tc->name, tc->mode, ref[i]) == 0;
    snprintf(name, sizeof(name), "cond_%s_built_n_exact", tc->name);
    check(name, st == ES_OK && have_ref && eng[i]->n_tokens == ref[i]->n_tokens, eng[i]->n_tokens, ref[i]->n_tokens);
    double ef = es_max_abs_diff(eng[i]->film, ref[i]->film, 4096);
    double ek = es_max_abs_diff(&eng[i]->k[0][0][0], &ref[i]->k[0][0][0], 4 * 68 * 16);
    double ev = es_max_abs_diff(&eng[i]->v[0][0][0], &ref[i]->v[0][0][0], 4 * 68 * 16);
    snprintf(name, sizeof(name), "cond_%s_film", tc->name); check_le(name, ef, tol.film);
    snprintf(name, sizeof(name), "cond_%s_k", tc->name);    check_le(name, ek, tol.k);
    snprintf(name, sizeof(name), "cond_%s_v", tc->name);    check_le(name, ev, tol.v);
    note(file, K_FILM, ef); note(file, K_K, ek); note(file, K_V, ev);
  }

  /* 4. Streams: (a) reference conditioning with FZ off and on, (b) engine conditioning, and 6. bit
     determinism (20-engine 3.5): each case's FZ-off run is repeated from reset and must match, memcmp. */
  float *out = malloc(sizeof(float) * (size_t)N), *heads = malloc(sizeof(float) * (size_t)(N / 64) * 2);
  float *out_a = malloc(sizeof(float) * (size_t)N), *heads_a = malloc(sizeof(float) * (size_t)(N / 64) * 2);
  for (int i = 0; i < 4; i++) {
    const char *cs = CASES[i].name;
    snprintf(name, sizeof(name), "out.%s", cs);
    float *out_ref = need_f32(&ct, name, (size_t)N);
    snprintf(name, sizeof(name), "heads.%s", cs);
    float *heads_ref = need_f32(&ct, name, (size_t)(N / 64) * 2);
    if (out_ref && heads_ref) {
      for (int fz = 0; fz <= 1; fz++) {
        set_fz(fz);
        int fz_now = get_fz();
        int r = run_stream(m, ref[i], in, N, out, heads);
        set_fz(0);
        if (i == 0) {
          snprintf(name, sizeof(name), "fpcr_fz_is_%d_during_run", fz);
          check_true(name, fz_now == fz);
        }
        snprintf(name, sizeof(name), "%s_refcond_fz%d", cs, fz);
        check_run(file, name, out, heads, out_ref, heads_ref, N, r, &tol);
        if (fz == 0) {                             /* kept for the determinism check below */
          memcpy(out_a, out, sizeof(float) * (size_t)N);
          memcpy(heads_a, heads, sizeof(float) * (size_t)(N / 64) * 2);
        }
      }
      int r = run_stream(m, eng[i], in, N, out, heads);
      snprintf(name, sizeof(name), "%s_engcond", cs);
      check_run(file, name, out, heads, out_ref, heads_ref, N, r, &tol);
      run_stream(m, ref[i], in, N, out, heads);    /* 6. the FZ-off run again, from reset */
      snprintf(name, sizeof(name), "determinism_%s_twice_memcmp", cs);
      check_true(name, memcmp(out, out_a, sizeof(float) * (size_t)N) == 0 &&
                       memcmp(heads, heads_a, sizeof(float) * (size_t)(N / 64) * 2) == 0);
    }
    free(out_ref); free(heads_ref);
  }

  /* 5. Debug taps (short file only). */
  if (file == 0) check_dbg(m, &ct, ref[0], in, &tol);

  /* 6b. Short file only: the whole person path (enroll, cond, stream) run as usual, then again with the
     process in the Darwin background state, which macOS keeps on the efficiency cores (what
     `taskpolicy -c background` does), then the state is restored. Both runs must match bit for bit.
     E-1 found that Accelerate can round differently on an E-core when a buffer's alignment changes; a
     background-QoS thread alone was measured to stay on a performance core, so the whole process moves.
     The E-core run is about 3 to 5 times slower; the INFO line shows both times. */
  if (file == 0) {
    e2e_job pc = { m, enr_t, enr_o, e_t, e_o, in, mask_t, mask_o, Lt, Lo, N, NULL, NULL, NULL, NULL, NULL, 0, 0 };
    e2e_job ec = pc;
    run_e2e(&pc);
    int moved = setpriority(PRIO_DARWIN_PROCESS, 0, PRIO_DARWIN_BG) == 0;
    run_e2e(&ec);
    int restored = setpriority(PRIO_DARWIN_PROCESS, 0, 0) == 0;
    check_true("determinism_person_end_to_end_efficiency_cores_memcmp", moved && restored && e2e_same(&pc, &ec));
    printf("INFO end to end (enroll, cond, %d hops): as usual %.0f ms, efficiency cores %.0f ms\n", N / ES_HOP,
           pc.ms, ec.ms);
    e2e_free(&pc);
    e2e_free(&ec);
  }
  free(out); free(heads); free(out_a); free(heads_a);
  for (int i = 0; i < 4; i++) { es_cond_free(eng[i]); es_cond_free(ref[i]); }
  es_voiceprint_free(vt);
  es_voiceprint_free(vo);

  /* es_selftest on this file (the app runs it on the short one at every launch). */
  float sel_err = -1;
  st = es_selftest(m, buf, nbytes, &sel_err);
  snprintf(name, sizeof(name), "selftest_%s_ok", file == 0 ? "short" : "long");
  check(name, st == ES_OK, sel_err, tol.out);
out:
  free(enr_t); free(enr_o); free(e_t); free(e_o); free(in); free(tok_t); free(tok_o); free(g_t); free(g_o);
  es_container_close(&ct);
  free(buf);
  free(path);
}

/* es_selftest must refuse test vectors made for other weights, and es_embed_g must behave. */
static void test_selftest_and_embed(const es_model *m, const char *dir) {
  char *path = join(dir, "testvec_short.est");
  void *buf = NULL;
  size_t n = 0;
  if (es_read_file(path, &buf, &n) != ES_OK) { free(path); return; }
  es_model *syn = NULL;
  es_model_synthetic(7, 0, &syn);
  float e = 0;
  check_true("selftest_other_weights_refused", es_selftest(syn, buf, n, &e) == ES_ERR_SELFTEST);
  es_model_free(syn);
  check_true("selftest_truncated_is_format_error", es_selftest(m, buf, n - 1, &e) == ES_ERR_FORMAT);
  check_true("selftest_null_args", es_selftest(NULL, buf, n, &e) == ES_ERR_ARG &&
                                   es_selftest(m, NULL, n, &e) == ES_ERR_ARG);

  /* meta.tolerances values are pinned to contract 3.3. The JSON header is outside data_sha256, so a header
     that loosens a value still opens; es_tv_tolerances (this test) and es_selftest (the app) must refuse it. */
  const char *tamper[2][3] = { { "\"out\":0.0002", "\"out\":0.0009", "out" },
                               { "\"out_snr_db\":60.0", "\"out_snr_db\":10.0", "out_snr_db" } };
  for (int i = 0; i < 2; i++) {
    uint8_t *bad = patch_copy(buf, n, tamper[i][0], tamper[i][1]);
    char err[256] = "";
    es_container ctb;
    int opened = bad && es_container_open(bad, n, "EST1", &ctb, err, sizeof(err)) == ES_OK;
    es_tv_tol tt;
    int refused = opened && es_tv_tolerances(&ctb, &tt, err, sizeof(err)) == ES_ERR_FORMAT &&
                  strstr(err, tamper[i][2]) != NULL;
    if (opened) es_container_close(&ctb);
    int selftest_refused = bad && es_selftest(m, bad, n, &e) == ES_ERR_SELFTEST;
    char name[64];
    snprintf(name, sizeof(name), "tolerance_%s_loosened_refused", tamper[i][2]);
    check_true(name, opened && refused && selftest_refused);
    printf("  es_tv_tolerances said: %s\n", err);
    free(bad);
  }

  /* es_embed_g: unit-length g on the target clip; ENROLL_SHORT on a clip too short for 4 tokens. */
  es_container ct;
  if (es_container_open(buf, n, "EST1", &ct, NULL, 0) == ES_OK) {
    float *enr_t = es_tv_f32(&ct, "enr_t", 96000);
    float g[128], ss = 0;
    es_status st = enr_t ? es_embed_g(m, enr_t, 96000, g) : ES_ERR_ARG;
    for (int i = 0; i < 128; i++) ss += g[i] * g[i];
    check("embed_g_unit_length", st == ES_OK && fabsf(sqrtf(ss) - 1.0f) < 1e-5f, sqrtf(ss), 1);
    check_true("embed_g_short_clip", enr_t && es_embed_g(m, enr_t, 4000, g) == ES_ERR_ENROLL_SHORT);
    free(enr_t);
    es_container_close(&ct);
  }
  free(buf);
  free(path);
}

int main(int argc, char **argv) {
  const char *dir = argc > 1 ? argv[1] : "../ml/exports/random_v1";
  const char *threads = getenv("VECLIB_MAXIMUM_THREADS");
  printf("test_parity (package E-1) export %s, VECLIB_MAXIMUM_THREADS=%s, hop path %s\n", dir, threads ? threads : "(unset)", es_hop_path);
  if (!threads || strcmp(threads, "1") != 0)
    printf("NOTE BLAS may be multi-threaded; run through `make parity` for the determinism check\n");

  es_model *m = test_loader(dir);
  if (!m) {
    printf("test_parity: cannot load the model, %d passed, %d failed\n", g_pass, g_fail + 1);
    return 1;
  }
  test_file(m, dir, "testvec_short.est", 0);
  test_file(m, dir, "testvec_long.est", 1);
  test_selftest_and_embed(m, dir);
  es_model_free(m);

  /* Summary per tolerance key: worst error (out_snr_db: lowest SNR) against the file's limit. The margin
     is limit / worst for errors and worst - limit (dB) for the SNR. */
  printf("== summary per tolerance key (short | long)\n");
  for (int k = 0; k < K_COUNT; k++) {
    printf("SUMMARY %-10s", KEY_NAMES[k]);
    for (int f = 0; f < 2; f++) {
      if (!g_seen[f][k]) { printf(" | %-36s", "not measured (no such arrays)"); continue; }
      double w = g_worst[f][k], l = g_limit[f][k];
      if (k == K_SNR) printf(" | worst %9.2f dB limit %5.1f margin %+.1f dB", w, l, w - l);
      else if (w > 0) printf(" | worst %9.3g limit %7.3g margin %6.0fx", w, l, l / w);
      else            printf(" | worst %9.3g limit %7.3g margin  exact", w, l);
    }
    printf("\n");
  }
  printf("test_parity: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
