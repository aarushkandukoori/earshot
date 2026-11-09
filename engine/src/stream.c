/* stream.c - one instance of the model, run one 64-sample hop at a time (20-engine 6, contract 2.1,
 * 2.2 and 2.6).
 *
 * es_stream_new allocates everything (state and scratch) once; es_stream_hop never allocates, locks or
 * logs, so it may run on the real-time worker. es_stream_reset only zeroes memory (RT-safe).
 * es_stream_new and es_stream_free are non-RT.
 * One catch measured on the M4: the first BLAS call of each kind a thread makes (or its BLASSetThreading
 * call) allocates once for that thread inside Accelerate. The worker calls BLASSetThreading and warm_blas
 * at thread start (rt_thread.c, 20-engine 3.5), which take those allocations before any hop; after them
 * a hop makes 0 allocations on macOS and the iOS 26.0 simulator (make rt, make sim-rt). The iPhone's
 * Accelerate is checked by the lead's D68 malloc hook, not by these tests.
 *
 * One hop, in the exact order of 20-engine 6.3:
 *    1-3  shift the 64 new samples into the analysis history, window, FFT       (stft.c)
 *    4    features: 3 numbers per bin                                            (stft.c)
 *    5    band encoder: per band Linear + LayerNorm -> z [16][64]
 *    6    4 dual-path blocks: time LSTM per band (state kept across hops), then a BiLSTM across the
 *         16 bands; FiLM 0 and cross-attention after block 0, FiLM 1 after block 2. Each block's six
 *         16-row products go through es_linear_hop (nn.c): one cblas_sgemv per row by default, or
 *         cblas_sgemm on the input transposed with -DES_HOP_SGEMM_XT (E-3, opt-in)
 *    7    heads: p_target and p_self from the mean over bands
 *    8    decoder: per band LayerNorm, Linear, tanh, Linear, then 1.5 tanh -> 1,026 coefficients
 *    9    deep filter: 3 taps on bands 0-11 (bins 0-127), 1 tap above
 *   10    inverse FFT, synthesis window, overlap-add, emit 64 samples              (stft.c)
 *   11    NaN guard: a non-finite output resets the stream and outputs zeros
 *
 * The output is the offline model's output delayed by one hop (64 samples), contract 2.1.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

es_stream *es_stream_new(const es_model *m) {
  if (!m) return NULL;
  void *mem = NULL;
  if (posix_memalign(&mem, 64, sizeof(es_stream)) != 0) return NULL;
  es_stream *s = mem;
  memset(s, 0, sizeof(*s));                        /* state and scratch start at zero */
  s->model = m;
  if (es_fft_init(&s->fft) != ES_OK) {
    free(s);
    return NULL;
  }
  return s;
}

void es_stream_free(es_stream *s) {
  if (!s) return;
  es_fft_free(&s->fft);
  free(s);
}

/* Zero the state of contract 2.6 (analysis history, deep-filter history, time-LSTM states, overlap-add).
   Scratch is not touched: every hop writes it before reading it. RT-safe (memset only). */
void es_stream_reset(es_stream *s) {
  if (s) memset(&s->st, 0, sizeof(s->st));
}

/* Cross-attention after FiLM 0 (contract 2.2 row 6, 20-engine 6.4). z [16][64] is flattened k*64 + d.
   4 heads of 16: head h uses q[16h .. 16h+15] and the keys / values c->k[h], c->v[h] of the first
   n = c->n_tokens tokens. Plain loops on purpose: they read like the formula (8,704 MACs). */
void es_attention(const es_model *m, const es_cond *c, float *z, float *q, float *sc, float *a) {
  es_linear(z, 1, 1024, 64, m->att_q.w, m->att_q.b, q);
  int n = c->n_tokens;
  for (int h = 0; h < 4; h++) {
    const float *K = &c->k[h][0][0], *V = &c->v[h][0][0], *qh = q + 16 * h;
    float *s = sc + h * ES_MAX_TOKENS;
    float mx = -INFINITY, sum = 0.0f;
    for (int j = 0; j < n; j++) {                  /* scores q_h . K_h[j] / sqrt(16) */
      float d = 0.0f;
      for (int e = 0; e < 16; e++) d += qh[e] * K[j * 16 + e];
      s[j] = 0.25f * d;
      mx = fmaxf(mx, s[j]);
    }
    for (int j = 0; j < n; j++) {                  /* softmax, shifted by the max for safety */
      s[j] = expf(s[j] - mx);
      sum += s[j];
    }
    float inv = 1.0f / sum;
    for (int e = 0; e < 16; e++) {                 /* a[16h + e] = sum_j p_j V_h[j][e] */
      float acc = 0.0f;
      for (int j = 0; j < n; j++) acc += s[j] * V[j * 16 + e];
      a[16 * h + e] = acc * inv;
    }
  }
  vDSP_vadd(z, 1, m->att_o.b, 1, z, 1, 1024);      /* z += o.bias */
  es_linear_acc(a, 1, 64, 1024, m->att_o.w, z);   /* z += a * o.weight */
}

/* Deep filter and history update (contract 2.2 row 10, 20-engine 6.5).
   Band k's coefficients are [re: T x W][im: T x W], index tau*W + j. Tap tau multiplies the spectrum
   of tau hops ago: tau = 0 is this hop's X, tau = 1 is xhist[0], tau = 2 is xhist[1]. Bands 0-11
   (3 taps) cover bins 0-127 exactly, so the history only holds those bins. */
void es_deep_filter(const es_model *m, const float *coef, const float *Xr, const float *Xi,
                    float xhist[2][ES_DF_BINS][2], float *Yr, float *Yi) {
  for (int k = 0; k < 16; k++) {
    int s0 = m->band_start[k], W = m->band_width[k], T = m->band_taps[k];
    const float *cr = coef + m->coef_off[k], *ci = cr + W * T;
    for (int j = 0; j < W; j++) {
      int f = s0 + j;
      float yr = cr[j] * Xr[f] - ci[j] * Xi[f];   /* tau = 0: complex multiply C_0 * X */
      float yi = cr[j] * Xi[f] + ci[j] * Xr[f];
      for (int tau = 1; tau < T; tau++) {          /* only for f < 128 */
        float xr = xhist[tau - 1][f][0], xi = xhist[tau - 1][f][1];
        float ar = cr[tau * W + j], ai = ci[tau * W + j];
        yr += ar * xr - ai * xi;
        yi += ar * xi + ai * xr;
      }
      Yr[f] = yr;
      Yi[f] = yi;
    }
  }
  memcpy(xhist[1], xhist[0], sizeof(float) * ES_DF_BINS * 2);     /* t-1 becomes t-2 */
  for (int f = 0; f < ES_DF_BINS; f++) {
    xhist[0][f][0] = Xr[f];
    xhist[0][f][1] = Xi[f];
  }
}

/* z = z * (1 + gamma) + beta for the 1,024 values of one FiLM point (film + off = gamma, + 1024 = beta). */
static void film_apply(float *z, const float *film, int off) {
  const float *gamma = film + off, *beta = film + off + 1024;
  for (int i = 0; i < 1024; i++) z[i] = z[i] * (1.0f + gamma[i]) + beta[i];
}

/* Steps 6a and 6b for block b: time path then band path, both with a pre-LayerNorm and a residual.
   The six 16-row products use es_linear_hop / es_linear_acc_hop. By default that is es_linear_rt (one
   cblas_sgemv per row): 0 allocations on macOS and the iOS simulator, and what the app ships. Built with
   -DES_HOP_SGEMM_XT it is es_linear_xt (cblas_sgemm on the input transposed into s->xt, about 3x
   faster), which waits for a malloc count on the iPhone before it can become the default. The plain
   cblas_sgemm would malloc inside the hop (contract 4.1 forbids it). nn.c has the measurements. */
static void dual_path_block(es_stream *s, int b) {
  const es_model *m = s->model;
  float *z = &s->z[0][0], *zn = &s->zn[0][0], *xt = &s->xt[0][0];
  float *h = &s->st.h_t[b][0][0];

  /* 6a. Time path: one LSTM(64 -> 128) shared by the 16 bands, each band carrying its own (h, c). */
  es_layernorm(z, 16, &m->blk[b].tn, zn);
  es_linear_hop(zn, 16, 64, 512, m->blk[b].t.wih, m->blk[b].t.bias, &s->gates[0][0], xt);  /* x w_ih + bias */
  es_linear_acc_hop(h, 16, 128, 512, m->blk[b].t.whh, &s->gates[0][0], xt);              /* + h w_hh */
  es_lstm_cell(16, 128, &s->gates[0][0], h, &s->st.c_t[b][0][0], &s->tmp[0][0]);
  es_add_rows(z, 16, 64, m->blk[b].tf.b);                                               /* z += tf.bias */
  es_linear_acc_hop(h, 16, 128, 64, m->blk[b].tf.w, z, xt);                               /* z += h tf.w */

  /* 6b. Band path: a BiLSTM(64 -> 64 each way) over the 16 bands as one sequence, zero state every hop. */
  es_layernorm(z, 16, &m->blk[b].bn, zn);
  es_linear_hop(zn, 16, 64, 256, m->blk[b].b.wih, m->blk[b].b.bias, &s->gxf[0][0], xt);
  es_linear_hop(zn, 16, 64, 256, m->blk[b].b_rev.wih, m->blk[b].b_rev.bias, &s->gxb[0][0], xt);
  es_lstm_seq(&s->gxf[0][0], 256, 0, &s->bcat[0][0], 128, 0, 16, 1, 64, m->blk[b].b.whh, 0, s->lw);
  es_lstm_seq(&s->gxb[0][0], 256, 0, &s->bcat[0][64], 128, 0, 16, 1, 64, m->blk[b].b_rev.whh, 1, s->lw);
  es_add_rows(z, 16, 64, m->blk[b].bf.b);                                               /* z += bf.bias */
  es_linear_acc_hop(&s->bcat[0][0], 16, 128, 64, m->blk[b].bf.w, z, xt);                  /* z += [f|b] bf.w */
}

/* True if all n values are finite (not NaN, not +/-Inf). */
static int all_finite(const float *x, int n) {
  for (int i = 0; i < n; i++)
    if (!isfinite(x[i])) return 0;
  return 1;
}

int es_stream_hop_dbg(es_stream *s, const es_cond *c, const float in[ES_HOP], float out[ES_HOP],
                      float heads[2], const es_dbg *d) {
  /* Precondition (20-engine 6.3): a conditioning set with 4..68 tokens. Otherwise output silence and
     leave the state alone (an empty engine slot). */
  if (!s || !c || c->n_tokens < 4 || c->n_tokens > ES_MAX_TOKENS) {
    if (out) memset(out, 0, sizeof(float) * ES_HOP);
    if (heads) heads[0] = heads[1] = 0.0f;
    return 0;
  }
  const es_model *m = s->model;
  float *z = &s->z[0][0];

  /* 1-3. Shift in the new samples, window, FFT. */
  es_analysis_hop(&s->fft, m->wa, s->st.ana, in, s->frame, s->Xr, s->Xi);

  /* 4. Features. */
  es_features(s->Xr, s->Xi, s->feat, s->fwork);
  if (d && d->feat) memcpy(d->feat, s->feat, sizeof(s->feat));

  /* 5. Band encoder: band k reads its 3 W_k features, which start at feat + 3 * band_start[k]. */
  for (int k = 0; k < 16; k++) {
    es_linear(s->feat + 3 * m->band_start[k], 1, 3 * m->band_width[k], 64, m->enc_lin[k].w, m->enc_lin[k].b,
              s->z[k]);
    es_layernorm(s->z[k], 1, &m->enc_ln[k], s->z[k]);
  }
  if (d && d->z_enc) memcpy(d->z_enc, z, sizeof(s->z));

  /* 6. The four dual-path blocks, FiLM 0 + attention after block 0, FiLM 1 after block 2. */
  for (int b = 0; b < 4; b++) {
    dual_path_block(s, b);
    if (b == 0) {
      film_apply(z, c->film, 0);                   /* point 0: gamma at 0, beta at 1024 */
      es_attention(m, c, z, s->q, &s->sc[0][0], s->a);
    }
    if (b == 2) film_apply(z, c->film, 2048);      /* point 1: gamma at 2048, beta at 3072 */
    if (d && d->z_blk[b]) memcpy(d->z_blk[b], z, sizeof(s->z));
  }

  /* 7. Heads: mean over the 16 bands, Linear(64 -> 2), sigmoid. */
  for (int e = 0; e < 64; e++) {
    float sum = 0.0f;
    for (int k = 0; k < 16; k++) sum += s->z[k][e];
    s->mu[e] = sum / 16.0f;
  }
  es_linear(s->mu, 1, 64, 2, m->heads.w, m->heads.b, s->h2);
  heads[0] = 1.0f / (1.0f + expf(-s->h2[0]));
  heads[1] = 1.0f / (1.0f + expf(-s->h2[1]));

  /* 8. Decoder: LayerNorm and Linear(64 -> 128) per band, one tanh over all 2,048 values, then
     Linear(128 -> 2 W_k T_k) per band, one tanh over all 1,026 values, times 1.5. */
  for (int k = 0; k < 16; k++) {
    es_layernorm(s->z[k], 1, &m->dec[k].ln, s->zn[k]);
    es_linear(s->zn[k], 1, 64, 128, m->dec[k].fc1.w, m->dec[k].fc1.b, s->hid[k]);
  }
  int n_hid = 16 * 128, n_coef = ES_COEF_TOTAL;
  vvtanhf(&s->hid[0][0], &s->hid[0][0], &n_hid);
  for (int k = 0; k < 16; k++) {
    int len = 2 * m->band_width[k] * m->band_taps[k];
    es_linear(s->hid[k], 1, 128, len, m->dec[k].fc2.w, m->dec[k].fc2.b, s->coef + m->coef_off[k]);
  }
  vvtanhf(s->coef, s->coef, &n_coef);
  float bound = 1.5f;                              /* mask_bound: every coefficient within (-1.5, 1.5) */
  vDSP_vsmul(s->coef, 1, &bound, s->coef, 1, ES_COEF_TOTAL);
  if (d && d->coef) memcpy(d->coef, s->coef, sizeof(s->coef));

  /* 9. Deep filter and history update. */
  es_deep_filter(m, s->coef, s->Xr, s->Xi, s->st.xhist, s->Yr, s->Yi);

  /* 10. Inverse FFT, synthesis window, overlap-add, emit 64 samples. */
  es_synthesis_hop(&s->fft, m->ws, s->Yr, s->Yi, s->frame, s->st.ola, out);

  /* 11. NaN guard: any non-finite output resets the stream; this hop outputs silence. */
  if (!all_finite(out, ES_HOP) || !all_finite(heads, 2)) {
    es_stream_reset(s);
    memset(out, 0, sizeof(float) * ES_HOP);
    heads[0] = heads[1] = 0.0f;
    return 1;
  }
  return 0;
}

int es_stream_hop(es_stream *s, const es_cond *c, const float in[ES_HOP], float out[ES_HOP], float heads[2]) {
  return es_stream_hop_dbg(s, c, in, out, heads, NULL);
}
