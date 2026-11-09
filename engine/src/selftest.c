/* selftest.c - es_selftest: the launch-time check that this build computes what PyTorch computes
 * (20-engine 13, contract 3.3). Also the small test-vector helpers that test_parity reuses.
 *
 * The app calls es_selftest once per launch with testvec_short.est. It fails fast:
 *   1. open the .est (format, version, config hash), then meta.weights_sha256 must equal the loaded
 *      model's hash, else "test vectors are for other weights"
 *   2. es_vad(enr_t) equals vad_t exactly
 *   3. enroll the target (enr_t, mask_t, e_t) and the own voice (enr_o, mask_o, e_o); tokens and g match
 *   4. es_cond_build(PERSON, target, own, 1) matches cond.person (n exact; film, k, v)
 *   5. a fresh stream runs all N / 64 hops of `in` with that conditioning; out.person and heads.person
 *      match, and the output SNR is at least meta.tolerances.out_snr_db
 * Threads: any non-RT thread (allocates). About 0.1 s on the phone (estimate, contract 3.3).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

/* The exact key set of meta.tolerances (contract 3.3), in the order of es_tv_tol's fields. */
static const char *TOL_KEYS[10] = { "vad", "tokens", "g", "film", "k", "v", "dbg", "out", "out_snr_db", "heads" };

static void set_err(char *err, size_t errlen, const char *msg, const char *key) {
  if (!err || errlen == 0) return;
  size_t n = 0;
  for (const char *p = msg; *p && n + 1 < errlen; p++) err[n++] = *p;
  for (const char *p = key ? key : ""; *p && n + 1 < errlen; p++) err[n++] = *p;
  err[n] = '\0';
}

es_status es_tv_tolerances(const es_container *ct, es_tv_tol *tol, char *err, size_t errlen) {
  if (!ct || !tol) return ES_ERR_ARG;
  const es_json *j = &ct->j;
  int obj = es_json_get_path(j, 0, "meta.tolerances");
  if (obj < 0 || j->tok[obj].type != ES_JSON_OBJECT) {
    set_err(err, errlen, "meta.tolerances: missing", NULL);
    return ES_ERR_FORMAT;
  }
  /* Every key in the file must be one of the 10 (an extra key fails). */
  int key = obj + 1;
  for (int c = 0; c < j->tok[obj].size; c++) {
    int known = 0;
    for (int i = 0; i < 10; i++) known |= es_json_str_eq(j, key, TOL_KEYS[i]);
    if (!known) {
      set_err(err, errlen, "meta.tolerances: unexpected key", NULL);
      return ES_ERR_FORMAT;
    }
    key = j->tok[key].next;
  }
  /* Every one of the 10 must be present as a finite number (a missing key fails, never a default). */
  double *field[10] = { &tol->vad, &tol->tokens, &tol->g, &tol->film, &tol->k, &tol->v, &tol->dbg, &tol->out,
                        &tol->out_snr_db, &tol->heads };
  for (int i = 0; i < 10; i++) {
    if (es_json_num(j, es_json_get(j, obj, TOL_KEYS[i]), field[i]) != 0) {
      set_err(err, errlen, "meta.tolerances: missing or not a number: ", TOL_KEYS[i]);
      return ES_ERR_FORMAT;
    }
  }
  if (j->tok[obj].size != 10) {                    /* catches a key written twice */
    set_err(err, errlen, "meta.tolerances: not exactly 10 keys", NULL);
    return ES_ERR_FORMAT;
  }

  /* Every value must equal the contract 3.3 number. The JSON header is not covered by data_sha256, so
     without this an exporter bug or an edited header could loosen the gate and let wrong outputs pass.
     out and out_snr_db depend on the file: short (N = 32,000 samples) or long (N = 160,000). */
  const es_array *in = es_container_find(ct, "in");
  int N = (in && in->ndim == 1) ? in->shape[0] : -1;
  double out_lim, snr_lim;
  if (N == 32000) {
    out_lim = 2e-4; snr_lim = 60.0;                /* testvec_short.est */
  } else if (N == 160000) {
    out_lim = 1e-3; snr_lim = 50.0;                /* testvec_long.est */
  } else {
    set_err(err, errlen, "in: N must be 32000 (short file) or 160000 (long file)", NULL);
    return ES_ERR_FORMAT;
  }
  const double want[10] = { 0.0, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, out_lim, snr_lim, 1e-3 };
  for (int i = 0; i < 10; i++) {
    /* Equal up to how the number was printed (1e-9 relative): "0.0001" and "1e-04" both pass. */
    if (fabs(*field[i] - want[i]) > 1e-9 * fabs(want[i])) {
      set_err(err, errlen, "meta.tolerances: value differs from contract 3.3: ", TOL_KEYS[i]);
      return ES_ERR_FORMAT;
    }
  }
  return ES_OK;
}

/* A malloc'd float copy (memcpy, so an unaligned caller buffer is fine). */
float *es_tv_f32(const es_container *ct, const char *name, size_t count) {
  const es_array *a = es_container_find(ct, name);
  if (!a || a->dtype != ES_DT_F32 || a->nbytes != count * sizeof(float)) return NULL;
  float *x = malloc(count > 0 ? a->nbytes : 1);
  if (x) memcpy(x, a->data, a->nbytes);
  return x;
}

double es_max_abs_diff(const float *x, const float *ref, size_t n) {
  double worst = 0.0;
  for (size_t i = 0; i < n; i++) {
    double d = fabs((double)x[i] - (double)ref[i]);
    if (!(d <= worst)) worst = isnan(d) ? INFINITY : d;   /* NaN counts as the worst possible error */
  }
  return worst;
}

double es_snr_db(const float *x, const float *ref, size_t n) {
  double sig = 0.0, err = 0.0;
  for (size_t i = 0; i < n; i++) {
    double d = (double)x[i] - (double)ref[i];
    sig += (double)ref[i] * ref[i];
    err += d * d;
  }
  if (isnan(err)) return -INFINITY;
  if (err == 0.0) return INFINITY;                 /* a zero error passes any SNR limit */
  return 10.0 * log10(sig / err);
}

/* Everything es_selftest allocates, so one cleanup path frees it all. */
typedef struct {
  es_container ct;
  int ct_open;
  float *enr_t, *enr_o, *e_t, *e_o, *tok_t, *g_t, *tok_o, *g_o, *film, *k, *v, *in, *out_ref, *heads_ref, *out;
  uint8_t *vad;
  es_voiceprint *vt, *vo;
  es_cond *cond;
  es_stream *s;
} selftest_mem;

static void selftest_free(selftest_mem *t) {
  float *bufs[] = { t->enr_t, t->enr_o, t->e_t, t->e_o, t->tok_t, t->g_t, t->tok_o, t->g_o, t->film, t->k, t->v,
                    t->in, t->out_ref, t->heads_ref, t->out };
  for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) free(bufs[i]);
  free(t->vad);
  es_voiceprint_free(t->vt);
  es_voiceprint_free(t->vo);
  es_cond_free(t->cond);
  es_stream_free(t->s);
  if (t->ct_open) es_container_close(&t->ct);
}

/* The u8 array `name` with exactly `count` entries, or NULL (points into the caller's buffer). */
static const uint8_t *tv_u8(const es_container *ct, const char *name, size_t count) {
  const es_array *a = es_container_find(ct, name);
  if (!a || a->dtype != ES_DT_U8 || a->nbytes != count) return NULL;
  return a->data;
}

/* Rows of a 2-D f32 array [rows, cols], or -1 if it is missing or has another width. */
static int tv_rows(const es_container *ct, const char *name, int cols) {
  const es_array *a = es_container_find(ct, name);
  if (!a || a->dtype != ES_DT_F32 || a->ndim != 2 || a->shape[1] != cols) return -1;
  return a->shape[0];
}

/* Length of a 1-D array, or -1. */
static int tv_len(const es_container *ct, const char *name) {
  const es_array *a = es_container_find(ct, name);
  if (!a || a->ndim != 1) return -1;
  return a->shape[0];
}

static es_status selftest_run(selftest_mem *t, const es_model *m, const void *est, size_t nbytes, float *max_err) {
  /* 1. Open, and require test vectors made from these exact weights. */
  es_status st = es_container_open(est, nbytes, "EST1", &t->ct, NULL, 0);
  if (st != ES_OK) return st;
  t->ct_open = 1;
  const es_container *ct = &t->ct;
  es_tv_tol tol;
  if (es_tv_tolerances(ct, &tol, NULL, 0) != ES_OK) return ES_ERR_SELFTEST;
  if (!es_json_str_eq(&ct->j, es_json_get_path(&ct->j, 0, "meta.weights_sha256"), es_model_data_hash(m)))
    return ES_ERR_SELFTEST;                        /* "test vectors are for other weights" */

  /* Inputs. */
  int Lt = tv_len(ct, "enr_t"), Lo = tv_len(ct, "enr_o"), N = tv_len(ct, "in");
  int nt = tv_rows(ct, "tok_t", 64), no = tv_rows(ct, "tok_o", 64);
  if (Lt <= 0 || Lo <= 0 || N <= 0 || N % ES_HOP != 0 || nt < 0 || no < 0) return ES_ERR_SELFTEST;
  int Ft = es_enroll_num_frames(Lt), Fo = es_enroll_num_frames(Lo);
  const uint8_t *mask_t = tv_u8(ct, "mask_t", (size_t)Ft), *mask_o = tv_u8(ct, "mask_o", (size_t)Fo);
  const uint8_t *vad_t = tv_u8(ct, "vad_t", (size_t)Ft);
  t->enr_t = es_tv_f32(ct, "enr_t", (size_t)Lt);
  t->enr_o = es_tv_f32(ct, "enr_o", (size_t)Lo);
  t->e_t = es_tv_f32(ct, "e_t", ES_WESPK_DIM);
  t->e_o = es_tv_f32(ct, "e_o", ES_WESPK_DIM);
  t->tok_t = es_tv_f32(ct, "tok_t", (size_t)nt * 64);
  t->g_t = es_tv_f32(ct, "g_t", ES_G_DIM);
  t->tok_o = es_tv_f32(ct, "tok_o", (size_t)no * 64);
  t->g_o = es_tv_f32(ct, "g_o", ES_G_DIM);
  t->film = es_tv_f32(ct, "cond.person.film", 4096);
  t->k = es_tv_f32(ct, "cond.person.k", 4 * 68 * 16);
  t->v = es_tv_f32(ct, "cond.person.v", 4 * 68 * 16);
  t->in = es_tv_f32(ct, "in", (size_t)N);
  t->out_ref = es_tv_f32(ct, "out.person", (size_t)N);
  t->heads_ref = es_tv_f32(ct, "heads.person", (size_t)(N / ES_HOP) * 2);
  const es_array *n_arr = es_container_find(ct, "cond.person.n");
  if (!mask_t || !mask_o || !vad_t || !t->enr_t || !t->enr_o || !t->e_t || !t->e_o || !t->tok_t || !t->g_t ||
      !t->tok_o || !t->g_o || !t->film || !t->k || !t->v || !t->in || !t->out_ref || !t->heads_ref || !n_arr ||
      n_arr->dtype != ES_DT_I32 || n_arr->nbytes != 4)
    return ES_ERR_SELFTEST;
  int32_t n_ref;
  memcpy(&n_ref, n_arr->data, 4);

  /* 2. The energy VAD, exactly. */
  t->vad = malloc((size_t)Ft);
  if (!t->vad) return ES_ERR_NOMEM;
  es_vad(t->enr_t, Lt, t->vad);
  int mismatches = 0;
  for (int f = 0; f < Ft; f++) mismatches += t->vad[f] != vad_t[f];
  if (mismatches > tol.vad) return ES_ERR_SELFTEST;

  /* 3. Enrollment of the target and the own voice with the masks from the file. */
  t->vt = es_voiceprint_new();
  t->vo = es_voiceprint_new();
  if (!t->vt || !t->vo) return ES_ERR_NOMEM;
  if (es_enroll(m, t->enr_t, Lt, mask_t, ES_ROLE_TARGET, t->e_t, t->vt) != ES_OK) return ES_ERR_SELFTEST;
  if (es_enroll(m, t->enr_o, Lo, mask_o, ES_ROLE_OWN, t->e_o, t->vo) != ES_OK) return ES_ERR_SELFTEST;
  if (t->vt->n_tok != nt || t->vo->n_tok != no) return ES_ERR_SELFTEST;
  if (es_max_abs_diff(&t->vt->tokens[0][0], t->tok_t, (size_t)nt * 64) > tol.tokens ||
      es_max_abs_diff(&t->vo->tokens[0][0], t->tok_o, (size_t)no * 64) > tol.tokens ||
      es_max_abs_diff(t->vt->g, t->g_t, ES_G_DIM) > tol.g || es_max_abs_diff(t->vo->g, t->g_o, ES_G_DIM) > tol.g)
    return ES_ERR_SELFTEST;

  /* 4. The person conditioning set. */
  t->cond = es_cond_new();
  if (!t->cond) return ES_ERR_NOMEM;
  if (es_cond_build(m, ES_MODE_PERSON, t->vt, t->vo, 1, t->cond) != ES_OK) return ES_ERR_SELFTEST;
  if (t->cond->n_tokens != n_ref || es_max_abs_diff(t->cond->film, t->film, 4096) > tol.film ||
      es_max_abs_diff(&t->cond->k[0][0][0], t->k, 4 * 68 * 16) > tol.k ||
      es_max_abs_diff(&t->cond->v[0][0][0], t->v, 4 * 68 * 16) > tol.v)
    return ES_ERR_SELFTEST;

  /* 5. Stream every hop of `in` from a fresh state with the engine-built conditioning. */
  t->s = es_stream_new(m);
  t->out = malloc(sizeof(float) * (size_t)N);
  if (!t->s || !t->out) return ES_ERR_NOMEM;
  double heads_err = 0.0;
  for (int j = 0; j < N / ES_HOP; j++) {
    float h[2];
    es_stream_hop(t->s, t->cond, t->in + (size_t)j * ES_HOP, t->out + (size_t)j * ES_HOP, h);
    heads_err = fmax(heads_err, es_max_abs_diff(h, t->heads_ref + (size_t)j * 2, 2));
  }
  double out_err = es_max_abs_diff(t->out, t->out_ref, (size_t)N);
  if (max_err) *max_err = (float)out_err;
  if (!(out_err <= tol.out) || !(es_snr_db(t->out, t->out_ref, (size_t)N) >= tol.out_snr_db) ||
      !(heads_err <= tol.heads))
    return ES_ERR_SELFTEST;
  return ES_OK;
}

es_status es_selftest(const es_model *m, const void *est, size_t nbytes, float *max_err_out) {
  if (max_err_out) *max_err_out = INFINITY;        /* until the stream has run */
  if (!m || !est) return ES_ERR_ARG;
  selftest_mem t;
  memset(&t, 0, sizeof(t));
  es_status st = selftest_run(&t, m, est, nbytes, max_err_out);
  selftest_free(&t);
  /* 20-engine 13: the result is ES_OK or ES_ERR_SELFTEST, apart from the format / version / config codes
     of opening the file. Running out of memory is reported as a failed self-test too. */
  if (st == ES_ERR_NOMEM) st = ES_ERR_SELFTEST;
  return st;
}
