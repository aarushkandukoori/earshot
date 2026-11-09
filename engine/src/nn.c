/* nn.c - neural-network kernels: Linear, LayerNorm, LSTM cell and LSTM sequence (20-engine 3.3, 6.2).
 *
 * Every function is float32 and takes all its buffers from the caller; any thread may call them.
 * es_linear and es_linear_acc with M >= 2 call cblas_sgemm in its plain form, which mallocs inside
 * Accelerate (measured on the M4), so only enrollment (non-RT) uses them that way. The hop's 16-row
 * products go through es_linear_hop / es_linear_acc_hop, which pick one of two paths at compile time:
 *   default            es_linear_rt (one cblas_sgemv per row): 0 allocations measured on macOS and on
 *                      the iOS 26.0 simulator; this is what the app ships
 *   -DES_HOP_SGEMM_XT  es_linear_xt (cblas_sgemm on X transposed, about 3x faster): 0 allocations on
 *                      macOS, but the unsplit form allocated on the iOS simulator (see es_linear_acc_xt)
 * The iPhone's Accelerate is not measured yet for either path (D68 malloc hook, lead).
 * Implements contract 2.2 (band encoder, LSTM time and band paths, decoder) and 2.4 (token encoder).
 *
 * Layout rules (contract 3.2):
 *   - Linear weights are stored [in][out] row-major, so Y = X * W needs no transpose.
 *   - LSTM weights are w_ih [in][4H], w_hh [H][4H], bias [4H] = bias_ih + bias_hh, gates in the
 *     order i (input), f (forget), g (cell candidate), o (output), like PyTorch.
 *
 * Sigmoid is computed as 0.5 * tanh(0.5 x) + 0.5, which is exactly the same function, so one vvtanhf
 * call handles all four gates of every row at once (20-engine 3.4).
 */
#include <math.h>
#include <string.h>

#include "internal.h"

/* Y[M][out] = X[M][in] * W[in][out] + b. One BLAS call: sgemv for a single row, sgemm otherwise. */
void es_linear(const float *X, int M, int in, int out, const float *W, const float *b, float *Y) {
  if (b) {
    for (int m = 0; m < M; m++) memcpy(Y + (size_t)m * out, b, sizeof(float) * (size_t)out);
  } else {
    memset(Y, 0, sizeof(float) * (size_t)M * (size_t)out);
  }
  es_linear_acc(X, M, in, out, W, Y);
}

/* Y[M][out] += X[M][in] * W[in][out]. BLAS computes alpha * X * W + beta * Y; here alpha = beta = 1.
   For M == 1, sgemv with CblasTrans treats W as a matrix of `in` rows and `out` columns and computes
   W^T * x, which is the row vector x * W. */
void es_linear_acc(const float *X, int M, int in, int out, const float *W, float *Y) {
  if (M == 1)
    cblas_sgemv(CblasRowMajor, CblasTrans, in, out, 1.0f, W, out, X, 1, 1.0f, Y, 1);
  else
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, out, in, 1.0f, X, in, W, out, 1.0f, Y, out);
}

/* Real-time twins of es_linear and es_linear_acc (E-1). They are the hop's DEFAULT path for its 16-row
   products (es_linear_hop below) and the reference the faster es_linear_xt is tested against.
   Same math, but always one cblas_sgemv per row, never cblas_sgemm.
   Why: contract 4.1 says es_stream_hop must not call malloc or free. On the M4 (it has SME), Accelerate's
   cblas_sgemm and vDSP_mmul malloc and free a 16 KB buffer inside every call with 2 or more rows, even
   after warm-up (measured 2026-11-07 with a counting malloc interposer: 24 allocations per hop when the
   hop used sgemm; BLASSetThreading single-threaded and the legacy CBLAS symbols did not stop it).
   cblas_sgemv made 0 allocations at every hop shape (macOS, and the iOS 26.0 simulator in the E-3 review
   and make sim-rt). The price: sgemm reads each weight matrix once, 16 sgemv
   calls read it 16 times. The whole hop got about 80 us slower on the shared M4 (median 267 -> 346 us,
   and 186 -> about 270 us in the machine's faster mode). Enrollment is not real-time, so it keeps
   es_linear (sgemm). The lead records this change to 20-engine 3.3 in DECISIONS. */
void es_linear_acc_rt(const float *X, int M, int in, int out, const float *W, float *Y) {
  for (int m = 0; m < M; m++)                    /* row m: Y[m] += X[m] * W, as in the M == 1 case above */
    cblas_sgemv(CblasRowMajor, CblasTrans, in, out, 1.0f, W, out, X + (size_t)m * in, 1, 1.0f,
                Y + (size_t)m * out, 1);
}

void es_linear_rt(const float *X, int M, int in, int out, const float *W, const float *b, float *Y) {
  if (b) {
    for (int m = 0; m < M; m++) memcpy(Y + (size_t)m * out, b, sizeof(float) * (size_t)out);
  } else {
    memset(Y, 0, sizeof(float) * (size_t)M * (size_t)out);
  }
  es_linear_acc_rt(X, M, in, out, W, Y);
}

/* The faster path (package E-3), OPT-IN since the E-3 fix pass: the hop uses it only when the engine is
   built with -DES_HOP_SGEMM_XT (and without -DES_HOP_SGEMV, which always wins). test_units checks it
   against es_linear_rt in every build, and make test-xt runs units, parity and rt on a build that uses it.

   Why opt-in: the E-3 review found that on the iOS 26.0 simulator runtime the single transposed
   cblas_sgemm with out = 512 mallocs and frees a 69,728-byte buffer in every call (2 + 1 per call; the
   time-LSTM products, 8 per hop: 16 allocations and 8 frees per hop), while out = 64, 256 and 384 made
   0. macOS showed 0 at every shape, so the macOS rt_guard tests could not see it. The fix pass splits
   every product into column blocks of at most XT_COLS = 256 (one sgemm per block, same ldb / ldc), which
   made 0 allocations on macOS and on the simulator (make sim-rt). Accelerate picks its buffers by shape
   and by build, so the iPhone may still differ: this path becomes the default only after the D68
   counting malloc hook shows 0 on the A18 Pro (DECISIONS L 2026-11-07 "Hop Linears").

   Same math, Y[M][out] (+)= X[M][in] * W[in][out], as cblas_sgemm calls with X handed to Accelerate
   already transposed (xt[in][M], "column k of X" stored contiguously) and CblasTrans on it.
   Measured on the M4 (2026-11-08, scratch kernel bench with the rt_guard interposer, 100 calls per hop
   shape after warm-up, on a P-core and an E-core; before the 256-column split):
     cblas_sgemm NoTrans/NoTrans (20-engine 3.3)   2 mallocs + 1 free per call   21 to 24 us per hop
     cblas_sgemm with X transposed (this)          0 allocations on macOS        22 to 27 us per hop
     one cblas_sgemv per row (es_linear_rt)        0 allocations                 67 to 75 us per hop
   ("per hop" =the 24 sixteen-row products of the 4 blocks, p50, load 4 to 6.5 on the shared Mac.
   Hand-written NEON kernels (4x16, 8x8 and 16x4 register blocks) took 160 to 240 us: the M4's
   Accelerate runs these on its matrix unit, which plain NEON cannot match. Other variants and their
   counts are in bench-results/2026-11-08-linear-ab.jsonl.) Whole hop, random_v1, bench hop_1inst p50:
   193 us with this path against 277 us with es_linear_rt (same Mac, same hour).
   Likely reason, not proven: the matrix unit multiplies one column of X by one row of W at a time, so
   with X in normal row order Accelerate first copies X into a malloc'd buffer in column order; given xt
   it can read the columns directly. The 24 transposes add about 1 to 2 us per hop in all.
   xt is caller scratch of at least M * in floats (64-byte aligned, like every buffer fed to Accelerate,
   DECISIONS L 2026-11-07).
   The first transposed sgemm a fresh thread makes allocates 2 x 8 bytes of per-thread state (E-3
   review; a tiny sgemv does not prevent it), so with this path the worker's warm_blas (rt_thread.c)
   must keep its tiny transposed sgemm before the first hop. */
enum { XT_COLS = 256 };   /* widest column block per sgemm call (512 allocated on the iOS simulator) */

void es_linear_acc_xt(const float *X, int M, int in, int out, const float *W, float *Y, float *xt) {
  vDSP_mtrans(X, 1, xt, 1, (vDSP_Length)in, (vDSP_Length)M);    /* xt[k][m] = X[m][k] */
  for (int j0 = 0; j0 < out; j0 += XT_COLS) {                    /* columns j0 .. j0 + n - 1 of Y and W */
    int n = out - j0 < XT_COLS ? out - j0 : XT_COLS;
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, n, in, 1.0f, xt, M, W + j0, out, 1.0f, Y + j0, out);
  }
}

void es_linear_xt(const float *X, int M, int in, int out, const float *W, const float *b, float *Y, float *xt) {
  if (b) {
    for (int m = 0; m < M; m++) memcpy(Y + (size_t)m * out, b, sizeof(float) * (size_t)out);
  } else {
    memset(Y, 0, sizeof(float) * (size_t)M * (size_t)out);
  }
  es_linear_acc_xt(X, M, in, out, W, Y, xt);
}

/* The hop's 16-row products (stream.c dual_path_block). One switch, here, picks the path for the whole
   engine: es_linear_rt by default (the app ships it), es_linear_xt with -DES_HOP_SGEMM_XT. ES_HOP_SGEMV
   (set by ios/project.yml) forces es_linear_rt even if ES_HOP_SGEMM_XT is also defined. xt is the
   transposed-input scratch the sgemm path needs; the sgemv path ignores it. */
#if defined(ES_HOP_SGEMM_XT) && !defined(ES_HOP_SGEMV)
const char *const es_hop_path = "xt";
void es_linear_hop(const float *X, int M, int in, int out, const float *W, const float *b, float *Y, float *xt) {
  es_linear_xt(X, M, in, out, W, b, Y, xt);
}
void es_linear_acc_hop(const float *X, int M, int in, int out, const float *W, float *Y, float *xt) {
  es_linear_acc_xt(X, M, in, out, W, Y, xt);
}
#else
const char *const es_hop_path = "sgemv";
void es_linear_hop(const float *X, int M, int in, int out, const float *W, const float *b, float *Y, float *xt) {
  (void)xt;
  es_linear_rt(X, M, in, out, W, b, Y);
}
void es_linear_acc_hop(const float *X, int M, int in, int out, const float *W, float *Y, float *xt) {
  (void)xt;
  es_linear_acc_rt(X, M, in, out, W, Y);
}
#endif

/* Y[m][j] += b[j] for every row m. Used before es_linear_acc for residual adds (20-engine 6.3). */
void es_add_rows(float *Y, int M, int n, const float *b) {
  for (int m = 0; m < M; m++) vDSP_vadd(Y + (size_t)m * n, 1, b, 1, Y + (size_t)m * n, 1, (vDSP_Length)n);
}

/* LayerNorm over C = 64 values per row, R rows, two passes like PyTorch: first the mean, then the mean
   of the squared deviations. y = (x - mean) / sqrt(var + 1e-5) * w + b. X may equal Y (in place). */
void es_layernorm(const float *X, int R, const es_ln *ln, float *Y) {
  for (int r = 0; r < R; r++) {
    const float *x = X + 64 * (size_t)r;
    float *y = Y + 64 * (size_t)r;
    float mean, var;
    vDSP_meanv(x, 1, &mean, 64);
    float neg_mean = -mean;
    vDSP_vsadd(x, 1, &neg_mean, y, 1, 64);       /* y = x - mean */
    vDSP_measqv(y, 1, &var, 64);                 /* var = mean(y^2) */
    float inv = 1.0f / sqrtf(var + ES_LN_EPS);
    vDSP_vsmul(y, 1, &inv, y, 1, 64);            /* y = (x - mean) / std */
    vDSP_vma(y, 1, ln->w, 1, ln->b, 1, y, 1, 64);/* y = y * w + b */
  }
}

/* R independent LSTM cells with hidden size H (20-engine 6.2).
   g holds R rows of 4H preactivations [i | f | g | o] and is overwritten.
   h and c (R x H) are updated in place; th (R x H) is scratch for tanh(c).
     i = sigmoid(gi), f = sigmoid(gf), gg = tanh(gg), o = sigmoid(go)
     c = f * c + i * gg;  h = o * tanh(c) */
void es_lstm_cell(int R, int H, float *g, float *h, float *c, float *th) {
  int n = R * 4 * H, m = R * H;
  float half = 0.5f;
  for (int r = 0; r < R; r++) {                 /* halve i, f and o (sigmoid via tanh); keep g */
    float *gr = g + (size_t)r * 4 * H;
    vDSP_vsmul(gr, 1, &half, gr, 1, (vDSP_Length)(2 * H));
    vDSP_vsmul(gr + 3 * H, 1, &half, gr + 3 * H, 1, (vDSP_Length)H);
  }
  vvtanhf(g, g, &n);                            /* one call for all gates of all rows */
  for (int r = 0; r < R; r++) {
    float *gr = g + (size_t)r * 4 * H, *cr = c + (size_t)r * H;
    for (int j = 0; j < H; j++) {
      float i = 0.5f * gr[j] + 0.5f;            /* sigmoid(x) = 0.5 tanh(x / 2) + 0.5 */
      float f = 0.5f * gr[H + j] + 0.5f;
      float gg = gr[2 * H + j];
      cr[j] = f * cr[j] + i * gg;
    }
  }
  vvtanhf(th, c, &m);                           /* th = tanh(c) */
  for (int r = 0; r < R; r++)
    for (int j = 0; j < H; j++)
      h[(size_t)r * H + j] = (0.5f * g[(size_t)r * 4 * H + 3 * H + j] + 0.5f) * th[(size_t)r * H + j];
}

/* B independent LSTM sequences of length T, each starting from zero state (20-engine 6.2).
   The input projections x * w_ih + bias are computed beforehand for every step: the 4H floats of
   step t, sequence b start at gx + t*gx_t + b*gx_b. The hidden output of step t, sequence b (H floats)
   is written to hs + t*hs_t + b*hs_b. reverse = 1 walks t = T-1 down to 0 (the backward direction).
   work holds B * 7 * H floats: h [B][H], c [B][H], gates [B][4H], th [B][H]. */
void es_lstm_seq(const float *gx, int gx_t, int gx_b, float *hs, int hs_t, int hs_b,
                 int T, int B, int H, const float *whh, int reverse, float *work) {
  float *h = work;
  float *c = h + (size_t)B * H;
  float *gates = c + (size_t)B * H;
  float *th = gates + (size_t)B * 4 * H;
  memset(h, 0, sizeof(float) * (size_t)B * H);
  memset(c, 0, sizeof(float) * (size_t)B * H);
  for (int step = 0; step < T; step++) {
    int t = reverse ? T - 1 - step : step;
    for (int b = 0; b < B; b++)                 /* this step's input projections, one row per sequence */
      memcpy(gates + (size_t)b * 4 * H, gx + (size_t)t * gx_t + (size_t)b * gx_b, sizeof(float) * 4 * (size_t)H);
    es_linear_acc(h, B, H, 4 * H, whh, gates);  /* gates += h * w_hh (the recurrent term) */
    es_lstm_cell(B, H, gates, h, c, th);
    for (int b = 0; b < B; b++)
      memcpy(hs + (size_t)t * hs_t + (size_t)b * hs_b, h + (size_t)b * H, sizeof(float) * (size_t)H);
  }
}
