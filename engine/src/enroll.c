/* enroll.c - enrollment: frame count and the energy VAD (contract 2.4, 20-engine 7.1).
 *
 * Package E-0 wrote es_enroll_num_frames and es_vad. E-1 adds the token encoder (es_tokens), es_enroll,
 * es_embed_g and the es_voiceprint_* calls here (20-engine 7.2, 7.3).
 * Threads: any non-RT thread (es_vad allocates two small arrays).
 *
 * The VAD must give exactly the same mask as ml/earshot/vad.py (bit for bit), so every energy is
 * computed in double with the samples summed in index order, and the percentile index uses integer
 * division (F - 1) / 10, which equals floor(0.1 * (F - 1)) for every F.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

/* Token frames of a clip of n samples: 0 below 576 samples (so F starts at 2), else (n - 64) / 512 + 1.
   Frame f covers samples 512 f - 448 .. 512 f + 63 of the clip (left-padded with 448 zeros). */
int es_enroll_num_frames(int n_samples) {
  if (n_samples < 576) return 0;
  return (n_samples - 64) / ES_TOK_HOP + 1;
}

/* qsort comparison for doubles, ascending. */
static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

/* Energy VAD (contract 2.4). Writes es_enroll_num_frames(n) bytes of 0 / 1 into mask_out.
     E[f]  = 10 log10(mean of the 512 squared samples of frame f + 1e-10)      (dB)
     P10   = the (F - 1) / 10 -th smallest E (a 10th percentile)
     thr   = max(P10 + 6, max(E) - 40)
     raw   = E > thr
     mask  = raw dilated by one frame on each side (out-of-range neighbours count as 0)
   With F = 0 it returns at once: nothing is read or written. If memory runs out the mask is all 0. */
void es_vad(const float *pcm, int n, uint8_t *mask_out) {
  int F = es_enroll_num_frames(n);
  if (F == 0 || pcm == NULL || mask_out == NULL) return;

  double *E = malloc(sizeof(double) * (size_t)F);
  double *sorted = malloc(sizeof(double) * (size_t)F);
  uint8_t *raw = malloc((size_t)F);
  if (!E || !sorted || !raw) {
    memset(mask_out, 0, (size_t)F);
    free(E); free(sorted); free(raw);
    return;
  }

  /* 1. Frame energies in dB, summed in index order in double. */
  for (int f = 0; f < F; f++) {
    double sum = 0.0;
    for (int i = 0; i < 512; i++) {
      long j = 512L * f + i - 448;               /* sample index in the clip; negative = left padding */
      double x = (j >= 0 && j < n) ? (double)pcm[j] : 0.0;
      sum += x * x;
    }
    E[f] = 10.0 * log10(sum / 512.0 + 1e-10);
  }

  /* 2. Threshold from the 10th percentile and the loudest frame. */
  memcpy(sorted, E, sizeof(double) * (size_t)F);
  qsort(sorted, (size_t)F, sizeof(double), cmp_double);
  double p10 = sorted[(F - 1) / 10];
  double emax = sorted[F - 1];
  double thr = fmax(p10 + 6.0, emax - 40.0);

  /* 3. Frames strictly above the threshold, then dilate by one frame each side. */
  for (int f = 0; f < F; f++) raw[f] = E[f] > thr ? 1 : 0;
  for (int f = 0; f < F; f++) {
    int left = f > 0 ? raw[f - 1] : 0;
    int right = f + 1 < F ? raw[f + 1] : 0;
    mask_out[f] = (uint8_t)(left | raw[f] | right);
  }

  free(E); free(sorted); free(raw);
}

/* ------------------------------------------------------------------------------------------------
 * Package E-1: token encoder, enrollment and voiceprints (contract 2.4, 20-engine 7.2 and 7.3).
 * ---------------------------------------------------------------------------------------------- */

#define ES_MAX_ENROLL_SAMPLES 960000   /* 60 s at 16 kHz (20-engine 7.2) */

/* Steps 4 and 6 of 20-engine 7.2 without the network: the frames with mask = 1 form one ordered list;
   it is cut into groups of 3 from its start (a final group of 1 or 2 is dropped), giving n_avail
   groups. If n_avail > limit, group (i * n_avail) / limit is kept for i < limit.
   Any nonzero mask byte counts as kept (a 0/255 or 0/2 mask means the same as 0/1), exactly like
   ml/earshot/vad.py token_groups (np.flatnonzero) and quality.py speech_seconds (np.count_nonzero). */
int es_token_groups(const uint8_t *mask, int F, int limit, int (*groups)[3], int *n_tok, int *kept) {
  int count = 0;
  for (int f = 0; f < F; f++) count += mask[f] != 0;
  int n_avail = count / 3;
  int n_keep = n_avail < limit ? n_avail : limit;
  /* The i-th group of the full list starts at the (3 i)-th kept frame; walk the mask once. */
  int g = 0, pos = 0, next = 0;                    /* next: index of the group we want next */
  for (int f = 0; f < F && g < n_keep; f++) {
    if (mask[f] == 0) continue;
    int group = pos / 3;                           /* this frame belongs to group `group` */
    if (n_avail > limit) next = (int)((int64_t)g * n_avail / limit);
    else next = g;
    if (group == next) {
      groups[g][pos % 3] = f;
      if (pos % 3 == 2) g++;                       /* group complete */
    }
    pos++;
  }
  if (n_tok) *n_tok = n_keep;
  if (kept) *kept = count;
  return n_avail;
}

/* The token encoder (20-engine 7.2). Work memory is allocated here and freed before returning. */
es_status es_tokens(const es_model *m, const float *pcm, int n, const uint8_t *mask, int limit,
                    float *tokens, int *n_tok, float g[ES_G_DIM], int *kept) {
  if (!m || !pcm || !mask || !tokens || !n_tok || !g || n <= 0 || n > ES_MAX_ENROLL_SAMPLES ||
      (limit != ES_MAX_TGT_TOKENS && limit != ES_MAX_OWN_TOKENS))
    return ES_ERR_ARG;
  int F = es_enroll_num_frames(n);

  /* Steps 4-5 first (they need only the mask): too few tokens fails before any network work. */
  int (*groups)[3] = malloc(sizeof(int[3]) * (size_t)limit);
  if (!groups) return ES_ERR_NOMEM;
  int nt = 0, kept_frames = 0;
  int n_avail = F > 0 ? es_token_groups(mask, F, limit, groups, &nt, &kept_frames) : 0;
  if (n_avail < 4) { free(groups); return ES_ERR_ENROLL_SHORT; }

  /* Work memory: z, zn, y [F][16][64]; gx, gxr [F][16][128]; LSTM work for B = max(16, F) rows;
     pooled pre-vectors [nt][1024] and the tokens [nt][64]. */
  size_t fz = (size_t)F * 1024, fg = (size_t)F * 2048;
  size_t lstm_rows = F > 16 ? (size_t)F : 16;
  /* 64-byte aligned like every other engine buffer: Accelerate's BLAS may pick a different kernel (and
     so a different rounding) for a differently aligned pointer, and plain malloc's alignment changes
     from run to run. Every piece below starts at a multiple of 16 floats, so all stay aligned. */
  void *mem = NULL;
  size_t work_floats = 3 * fz + 2 * fg + lstm_rows * 7 * 32 + (size_t)nt * 1024 + (size_t)nt * 64;
  float *work = posix_memalign(&mem, 64, sizeof(float) * work_floats) == 0 ? mem : NULL;
  _Alignas(64) es_fft fft;
  int fft_ok = es_fft_init(&fft) == ES_OK;
  if (!work || !fft_ok) {
    free(groups); free(work);
    if (fft_ok) es_fft_free(&fft);
    return ES_ERR_NOMEM;
  }
  float *z = work, *zn = z + fz, *y = zn + fz, *gx = y + fz, *gxr = gx + fg;
  float *lw = gxr + fg, *pre = lw + lstm_rows * 7 * 32, *tok = pre + (size_t)nt * 1024;

  /* Step 1. Frames of 512 at hop 512 from the clip left-padded with 448 zeros, window, FFT, features,
     and the shared band encoder of the separation model. */
  _Alignas(64) float frame[ES_NFFT];
  _Alignas(64) float Xr[ES_NBINS];
  _Alignas(64) float Xi[ES_NBINS];
  _Alignas(64) float feat[ES_NFEAT];
  _Alignas(64) float fwork[3 * ES_NBINS];
  for (int f = 0; f < F; f++) {
    for (int i = 0; i < ES_NFFT; i++) {
      long j = (long)ES_TOK_HOP * f + i - ES_ANA_RISE;     /* clip index; negative = padding */
      frame[i] = m->wa[i] * (j >= 0 ? pcm[j] : 0.0f);      /* j < n always, because of how F is computed */
    }
    es_rfft(&fft, frame, Xr, Xi);
    es_features(Xr, Xi, feat, fwork);
    for (int k = 0; k < 16; k++) {
      float *zk = z + (size_t)f * 1024 + 64 * k;
      es_linear(feat + 3 * m->band_start[k], 1, 3 * m->band_width[k], 64, m->enc_lin[k].w, m->enc_lin[k].b, zk);
      es_layernorm(zk, 1, &m->enc_ln[k], zk);
    }
  }
  es_fft_free(&fft);

  /* Step 2. Time path: BiLSTM(64 -> 32 each way) over the F frames, one sequence per band (B = 16).
     Row (f, k) of gx starts at f*16*128 + k*128; the outputs go to y[f][k][0..31] (fwd), [32..63] (rev). */
  int R = 16 * F;                                  /* rows: every (frame, band) pair */
  es_layernorm(z, R, &m->tok.tn, zn);
  es_linear(zn, R, 64, 128, m->tok.t.wih, m->tok.t.bias, gx);
  es_linear(zn, R, 64, 128, m->tok.t_rev.wih, m->tok.t_rev.bias, gxr);
  es_lstm_seq(gx, 16 * 128, 128, y, 16 * 64, 64, F, 16, 32, m->tok.t.whh, 0, lw);
  es_lstm_seq(gxr, 16 * 128, 128, y + 32, 16 * 64, 64, F, 16, 32, m->tok.t_rev.whh, 1, lw);
  es_add_rows(z, R, 64, m->tok.tf.b);
  es_linear_acc(y, R, 64, 64, m->tok.tf.w, z);     /* z += y tf.w + tf.b */

  /* Step 3. Band path: BiLSTM over the 16 bands, one sequence per frame (B = F). */
  es_layernorm(z, R, &m->tok.bn, zn);
  es_linear(zn, R, 64, 128, m->tok.b.wih, m->tok.b.bias, gx);
  es_linear(zn, R, 64, 128, m->tok.b_rev.wih, m->tok.b_rev.bias, gxr);
  es_lstm_seq(gx, 128, 16 * 128, y, 64, 16 * 64, 16, F, 32, m->tok.b.whh, 0, lw);
  es_lstm_seq(gxr, 128, 16 * 128, y + 32, 64, 16 * 64, 16, F, 32, m->tok.b_rev.whh, 1, lw);
  es_add_rows(z, R, 64, m->tok.bf.b);
  es_linear_acc(y, R, 64, 64, m->tok.bf.w, z);     /* z += y bf.w + bf.b */

  /* Steps 4 and 6. Each kept token's pre-vector is the mean of its 3 frames (1,024 floats each). */
  for (int i = 0; i < nt; i++) {
    const float *za = z + (size_t)groups[i][0] * 1024, *zb = z + (size_t)groups[i][1] * 1024;
    const float *zc = z + (size_t)groups[i][2] * 1024;
    float *p = pre + (size_t)i * 1024;
    for (int e = 0; e < 1024; e++) p[e] = (za[e] + zb[e] + zc[e]) / 3.0f;
  }

  /* Step 7. tokens = pre proj.w + proj.b; g = normalize(mean(tokens) g.w + g.b). */
  es_linear(pre, nt, 1024, 64, m->tok_proj.w, m->tok_proj.b, tok);
  _Alignas(64) float gbar[64];
  for (int e = 0; e < 64; e++) {
    float sum = 0.0f;
    for (int i = 0; i < nt; i++) sum += tok[(size_t)i * 64 + e];
    gbar[e] = sum / (float)nt;
  }
  _Alignas(64) float graw[ES_G_DIM];
  es_linear(gbar, 1, 64, ES_G_DIM, m->tok_g.w, m->tok_g.b, graw);
  double ss = 0.0;
  for (int e = 0; e < ES_G_DIM; e++) ss += (double)graw[e] * graw[e];
  float norm = (float)sqrt(ss);
  float inv = 1.0f / (norm > 1e-12f ? norm : 1e-12f);   /* PyTorch F.normalize: x / max(||x||, 1e-12) */
  for (int e = 0; e < ES_G_DIM; e++) g[e] = graw[e] * inv;

  memcpy(tokens, tok, sizeof(float) * (size_t)nt * 64);
  *n_tok = nt;
  if (kept) *kept = kept_frames;
  free(groups);
  free(work);
  return ES_OK;
}

/* A new, empty voiceprint. role = -1 marks it as not enrolled, so es_cond_build refuses it. */
es_voiceprint *es_voiceprint_new(void) {
  es_voiceprint *v = calloc(1, sizeof(es_voiceprint));
  if (v) v->role = -1;
  return v;
}

void es_voiceprint_free(es_voiceprint *v) { free(v); }

int es_voiceprint_num_tokens(const es_voiceprint *v) { return v ? v->n_tok : 0; }

void es_voiceprint_get_g(const es_voiceprint *v, float g_out[ES_G_DIM]) {
  if (!g_out) return;
  if (v) memcpy(g_out, v->g, sizeof(v->g));
  else memset(g_out, 0, sizeof(float) * ES_G_DIM);
}

int es_voiceprint_has_wespk(const es_voiceprint *v) { return v ? v->has_wespk : 0; }

float es_voiceprint_speech_seconds(const es_voiceprint *v) { return v ? v->speech_s : 0.0f; }

/* Enrollment (20-engine 7.3): token encoder with the given mask (or es_vad when mask is NULL), limit 48
   for a target and 16 for the user's own voice, plus a copy of the WeSpeaker vector if given.
   `out` is written only on success. */
es_status es_enroll(const es_model *m, const float *pcm, int n, const uint8_t *mask, es_role role,
                    const float *e_wespk, es_voiceprint *out) {
  if (!m || !pcm || !out || n <= 0 || n > ES_MAX_ENROLL_SAMPLES) return ES_ERR_ARG;
  if (role != ES_ROLE_TARGET && role != ES_ROLE_OWN) return ES_ERR_ARG;
  for (int i = 0; i < n; i++)
    if (!isfinite(pcm[i])) return ES_ERR_ARG;
  if (e_wespk)
    for (int i = 0; i < ES_WESPK_DIM; i++)
      if (!isfinite(e_wespk[i])) return ES_ERR_ARG;

  int F = es_enroll_num_frames(n);
  uint8_t *own_mask = NULL;
  if (!mask && F > 0) {                            /* no mask given: run the energy VAD */
    own_mask = malloc((size_t)F);
    if (!own_mask) return ES_ERR_NOMEM;
    es_vad(pcm, n, own_mask);
    mask = own_mask;
  }
  if (F == 0) {                                    /* under 576 samples: no frames, so no tokens */
    free(own_mask);
    return ES_ERR_ENROLL_SHORT;
  }

  es_voiceprint *v = calloc(1, sizeof(es_voiceprint));   /* built aside, copied to out on success */
  if (!v) { free(own_mask); return ES_ERR_NOMEM; }
  int limit = role == ES_ROLE_TARGET ? ES_MAX_TGT_TOKENS : ES_MAX_OWN_TOKENS;
  int kept = 0;
  es_status st = es_tokens(m, pcm, n, mask, limit, &v->tokens[0][0], &v->n_tok, v->g, &kept);
  if (st == ES_OK) {
    v->role = (int)role;
    v->speech_s = (float)kept * 0.032f;            /* frames with mask != 0, 32 ms each */
    v->has_wespk = e_wespk != NULL;
    if (e_wespk) memcpy(v->e, e_wespk, sizeof(v->e));   /* stored as given */
    *out = *v;
  }
  free(v);
  free(own_mask);
  return st;
}

/* The "locked" check (contract 2.4 step 8): es_vad mask, the target limit, returns only g. */
es_status es_embed_g(const es_model *m, const float *pcm, int n, float g_out[ES_G_DIM]) {
  if (!m || !pcm || !g_out || n <= 0 || n > ES_MAX_ENROLL_SAMPLES) return ES_ERR_ARG;
  for (int i = 0; i < n; i++)
    if (!isfinite(pcm[i])) return ES_ERR_ARG;
  int F = es_enroll_num_frames(n);
  if (F == 0) return ES_ERR_ENROLL_SHORT;
  uint8_t *mask = malloc((size_t)F);
  float *tok = malloc(sizeof(float) * ES_MAX_TGT_TOKENS * ES_TOK_DIM);
  if (!mask || !tok) { free(mask); free(tok); return ES_ERR_NOMEM; }
  es_vad(pcm, n, mask);
  int nt = 0;
  float g[ES_G_DIM];
  es_status st = es_tokens(m, pcm, n, mask, ES_MAX_TGT_TOKENS, tok, &nt, g, NULL);
  if (st == ES_OK) memcpy(g_out, g, sizeof(g));
  free(mask);
  free(tok);
  return st;
}
