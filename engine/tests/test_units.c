/* test_units.c - unit tests for the engine building blocks (20-engine 14.1). Needs no ML export.
 *
 * Package E-0 covers: config hash, windows, FFT, STFT streaming and offline, features, kernels
 * (Linear, LayerNorm, LSTM cell, LSTM sequence) against naive double C, energy VAD, json_min, the
 * container reader, rings (including the history ring under a racing writer), resampler (response from
 * impulses, block-size independence, need-driven interpolation bit-identical to one block, prefill
 * table), ramps, limiter, soft clip, the 270-tensor list and the synthetic models, defaults, status
 * strings and lock-free atomics.
 * Package E-1 added: the model-level all-pass row (the synthetic all-pass model streams the input back
 * delayed by 64 samples), stream determinism, the NaN guard, empty conditioning, attention and the deep
 * filter against naive double code, pooling and selection, and the enrollment and es_cond_build rules
 * (including "use_wespk with one vector missing gives both null").
 * Package E-2 added: the history ring's future-cursor rule, engine create rules and the P_min prefill,
 * the muted start state and fade-in, input taps, stats atomics, params clamping, the post formulas
 * (gate, duck, room envelope, AGC slew and target, bypass), cue beeps and the runaway guard. The
 * engine-level rows (alignment, gain cap, hand-off race, impulse delay, slack step) are in
 * tests/test_latency.c, the zero-allocation run in tests/test_rt.c.
 * E-2 part B added: the latency probe through a simulated echo loop (20-engine 12: known delays at 48 and
 * 44.1 kHz, noise only, the n_ok rule, the spread, the call rules) and the WAV reader and writer
 * (tools/wav.c, compiled into this test by the Makefile).
 * E-2 fix pass added: the gain cap on steady noise (step 5 equals the 20-engine 10 formula, and the gain
 * it applies moves at most 0.5 dB per hop, so it does not pump).
 * Package E-3 added: the hop's fast 16-row Linear (es_linear_xt, sgemm on X transposed) against its
 * reference es_linear_rt on every hop shape, and the probe's 300 Hz to 6 kHz GCC-PHAT band at -36 dBFS
 * noise against the full band (DECISIONS L 2026-11-08).
 * E-3 fix pass: es_linear_xt is opt-in (-DES_HOP_SGEMM_XT) and split into column blocks of <= 256, so
 * its row also covers a width that is not a multiple of 256; a new row checks that es_linear_hop (the
 * path this build compiled, printed as INFO) equals es_linear_xt or es_linear_rt exactly.
 *
 * Every check prints one line "PASS name value limit" or "FAIL name value limit"; the program exits 1
 * if any check failed. Run: make test (from engine/).
 */
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "internal.h"
#include "ring.h"
#include "wav.h"

/* ---------------------------------------------------------------------------------------------- */
/* Tiny test harness                                                                               */
/* ---------------------------------------------------------------------------------------------- */
static int g_pass = 0, g_fail = 0;

static void check(const char *name, int ok, double value, double limit) {
  printf("%s %s %.6g %.6g\n", ok ? "PASS" : "FAIL", name, value, limit);
  if (ok) g_pass++; else g_fail++;
}
/* value must be at most limit (NaN fails) */
static void check_le(const char *name, double value, double limit) { check(name, value <= limit, value, limit); }
/* exact yes / no checks print 1 or 0 */
static void check_true(const char *name, int ok) { check(name, ok, ok ? 1 : 0, 1); }

/* xorshift32 random numbers, deterministic */
static uint32_t g_rng = 2463534242u;
static uint32_t xs32(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }
static float urand(void) { return (float)((xs32() >> 8) * (1.0 / 16777216.0)) * 2.0f - 1.0f; }   /* [-1, 1) */
static void fill_rand(float *x, int n, float scale) { for (int i = 0; i < n; i++) x[i] = scale * urand(); }

static double max_abs_diff_fd(const float *a, const double *b, int n) {
  double m = 0;
  for (int i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (!(d <= m)) m = d; }
  return m;
}
static double max_abs_diff_ff(const float *a, const float *b, int n) {
  double m = 0;
  for (int i = 0; i < n; i++) { double d = fabs((double)a[i] - (double)b[i]); if (!(d <= m)) m = d; }
  return m;
}

/* ---------------------------------------------------------------------------------------------- */
/* Config hash (20-engine 4.2)                                                                     */
/* ---------------------------------------------------------------------------------------------- */
static void test_config(void) {
  char hex[65];
  es_sha256_hex(ES_CONFIG_CANON, strlen(ES_CONFIG_CANON), hex);
  check("config_canon_length", strlen(ES_CONFIG_CANON) == 470, (double)strlen(ES_CONFIG_CANON), 470);
  check_true("config_canon_sha256", strcmp(hex, ES_CONFIG_HASH_HEX) == 0);
}

/* ---------------------------------------------------------------------------------------------- */
/* Windows (20-engine 5.1)                                                                         */
/* ---------------------------------------------------------------------------------------------- */
static double hann_d(int n, int L) { return 0.5 - 0.5 * cos(2.0 * ES_PI * n / L); }

static void test_windows(void) {
  float wa[512], ws[512];
  es_make_windows(wa, ws);
  double err_formula = 0, err_tail = 0, err_sum = 0, min_wa = 1, max_ws = 0;
  for (int n = 0; n < 512; n++) {
    double a = n < 448 ? sqrt(hann_d(n, 896)) : sqrt(hann_d(n - 384, 128));
    double s = n < 384 ? 0.0 : hann_d(n - 384, 128) / a;
    err_formula = fmax(err_formula, fmax(fabs(wa[n] - a), fabs(ws[n] - s)));
    if (n >= 448) err_tail = fmax(err_tail, fabs(wa[n] - sqrt(hann_d(n - 384, 128))));
    if (n >= 384) { min_wa = fmin(min_wa, wa[n]); max_ws = fmax(max_ws, ws[n]); }
  }
  for (int i = 0; i < 64; i++) {                 /* two overlapping frames at hop 64 must sum to 1 */
    double sum = (double)wa[384 + i] * ws[384 + i] + (double)wa[448 + i] * ws[448 + i];
    err_sum = fmax(err_sum, fabs(sum - 1.0));
  }
  check_le("window_formulas", err_formula, 1e-7);
  check_le("window_wa_tail_is_sqrt_hann128", err_tail, 1e-7);
  check_le("window_hop64_sum_wa_ws", err_sum, 1e-6);
  check_le("window_min_wa_synth_span_0.0245", fabs(min_wa - 0.024541), 1e-5);
  check_le("window_max_ws", max_ws, 1.0);
}

/* ---------------------------------------------------------------------------------------------- */
/* FFT wrapper (20-engine 5.2): forward = torch.fft.rfft, inverse = torch.fft.irfft                */
/* ---------------------------------------------------------------------------------------------- */
static void naive_rfft(const float *x, double *Xr, double *Xi) {
  for (int k = 0; k < 257; k++) {
    double re = 0, im = 0;
    for (int n = 0; n < 512; n++) {
      double ang = 2.0 * ES_PI * (double)((k * n) % 512) / 512.0;
      re += x[n] * cos(ang);
      im -= x[n] * sin(ang);
    }
    Xr[k] = re; Xi[k] = im;
  }
}

/* irfft(n=512): x[n] = (1/512) (Y0 + Y256 (-1)^n + 2 sum_{k=1}^{255} Re(Y_k e^{+i 2 pi k n / 512})) */
static void naive_irfft(const float *Yr, const float *Yi, double *y) {
  for (int n = 0; n < 512; n++) {
    double s = Yr[0] + Yr[256] * ((n & 1) ? -1.0 : 1.0);
    for (int k = 1; k < 256; k++) {
      double ang = 2.0 * ES_PI * (double)((k * n) % 512) / 512.0;
      s += 2.0 * (Yr[k] * cos(ang) - Yi[k] * sin(ang));
    }
    y[n] = s / 512.0;
  }
}

static void test_fft(void) {
  es_fft f;
  check_true("fft_init", es_fft_init(&f) == ES_OK);
  float x[512], Xr[257], Xi[257], y[512];
  double Rr[257], Ri[257], yd[512];
  double err_fwd = 0, err_rt = 0, err_inv = 0;
  for (int trial = 0; trial < 10; trial++) {
    fill_rand(x, 512, 1.0f);
    es_rfft(&f, x, Xr, Xi);
    naive_rfft(x, Rr, Ri);
    double mx = 0, e = 0;
    for (int k = 0; k < 257; k++) {
      mx = fmax(mx, hypot(Rr[k], Ri[k]));
      e = fmax(e, hypot(Xr[k] - Rr[k], Xi[k] - Ri[k]));
    }
    err_fwd = fmax(err_fwd, e / mx);
    es_irfft(&f, Xr, Xi, y);                     /* round trip */
    for (int n = 0; n < 512; n++) err_rt = fmax(err_rt, fabs(y[n] - x[n]));
    /* inverse alone, with junk in Im of bins 0 and 256 that irfft must ignore */
    float Yr[257], Yi[257];
    fill_rand(Yr, 257, 1.0f); fill_rand(Yi, 257, 1.0f);
    es_irfft(&f, Yr, Yi, y);
    naive_irfft(Yr, Yi, yd);
    double my = 0, ei = 0;
    for (int n = 0; n < 512; n++) { my = fmax(my, fabs(yd[n])); ei = fmax(ei, fabs(y[n] - yd[n])); }
    err_inv = fmax(err_inv, ei / my);
  }
  check_le("fft_forward_vs_naive_dft_rel", err_fwd, 1e-6);
  check_le("fft_inverse_forward_roundtrip", err_rt, 1e-6);
  check_le("fft_inverse_vs_naive_irfft_rel", err_inv, 1e-6);

  /* torch scaling: rfft of ones has DC 512; a unit cosine at bin 5 gives X[5] = 256 */
  for (int n = 0; n < 512; n++) x[n] = 1.0f;
  es_rfft(&f, x, Xr, Xi);
  check_le("fft_torch_scale_dc_512", fabs(Xr[0] - 512.0), 1e-3);
  for (int n = 0; n < 512; n++) x[n] = (float)cos(2.0 * ES_PI * 5.0 * n / 512.0);
  es_rfft(&f, x, Xr, Xi);
  check_le("fft_torch_scale_bin5_256", fabs(Xr[5] - 256.0) + fabs(Xi[5]), 1e-3);
  check_le("fft_imag_dc_nyquist_zero", fabs(Xi[0]) + fabs(Xi[256]), 0.0);
  es_fft_free(&f);
}

/* ---------------------------------------------------------------------------------------------- */
/* STFT streaming and offline (contract 2.1, 20-engine 5.4)                                        */
/* ---------------------------------------------------------------------------------------------- */
/* A per-frame complex gain used to make the offline-vs-streaming test non-trivial. */
static void frame_gain(int t, int k, float *gr, float *gi) {
  *gr = 0.2f + 0.8f * cosf(0.01f * (float)(t * k));
  *gi = 0.5f * sinf(0.003f * (float)(t + k));
}

static void test_stft(void) {
  enum { T = 200, N = T * 64 };
  float wa[512], ws[512];
  es_make_windows(wa, ws);
  float *x = malloc(sizeof(float) * N), *ys = malloc(sizeof(float) * N), *yo = malloc(sizeof(float) * N);
  float *Or = malloc(sizeof(float) * T * 257), *Oi = malloc(sizeof(float) * T * 257);
  fill_rand(x, N, 0.5f);
  es_fft f;
  es_fft_init(&f);
  float ana[512] = { 0 }, ola[128] = { 0 }, frame[512], Xr[257], Xi[257];

  /* 1. All-pass (Y = X): the stream reproduces the input delayed by 64 samples. */
  double err_allpass = 0;
  for (int j = 0; j < T; j++) {
    es_analysis_hop(&f, wa, ana, x + 64 * j, frame, Xr, Xi);
    es_synthesis_hop(&f, ws, Xr, Xi, frame, ola, ys + 64 * j);
  }
  for (int n = 64; n < N; n++) err_allpass = fmax(err_allpass, fabs(ys[n] - x[n - 64]));
  for (int n = 0; n < 64; n++) err_allpass = fmax(err_allpass, fabs(ys[n]));   /* hop 0 emits silence */
  check_le("stft_stream_allpass_delay64", err_allpass, 1e-5);

  /* 2. With a per-frame gain: y_stream[n + 64] = y_offline[n]; frames equal the offline STFT. */
  memset(ana, 0, sizeof(ana));
  memset(ola, 0, sizeof(ola));
  check_true("stft_offline_ok", es_stft_offline(wa, x, N, Or, Oi) == ES_OK);
  double err_frames = 0;
  for (int t = 0; t < T; t++) {
    es_analysis_hop(&f, wa, ana, x + 64 * t, frame, Xr, Xi);
    for (int k = 0; k < 257; k++) {
      err_frames = fmax(err_frames, fabs(Xr[k] - Or[t * 257 + k]) + fabs(Xi[k] - Oi[t * 257 + k]));
      float gr, gi;
      frame_gain(t, k, &gr, &gi);
      float yr = gr * Xr[k] - gi * Xi[k], yi = gr * Xi[k] + gi * Xr[k];
      Xr[k] = yr; Xi[k] = yi;
      float orr = Or[t * 257 + k], oii = Oi[t * 257 + k];   /* same gain on the offline frames */
      Or[t * 257 + k] = gr * orr - gi * oii;
      Oi[t * 257 + k] = gr * oii + gi * orr;
    }
    es_synthesis_hop(&f, ws, Xr, Xi, frame, ola, ys + 64 * t);
  }
  check_true("istft_offline_ok", es_istft_offline(ws, Or, Oi, T, yo) == ES_OK);
  double err_off = 0;
  for (int n = 0; n + 64 < N; n++) err_off = fmax(err_off, fabs(ys[n + 64] - yo[n]));
  check_le("stft_stream_frames_equal_offline", err_frames, 1e-5);
  check_le("stft_stream_vs_offline_shift64", err_off, 1e-5);

  es_fft_free(&f);
  free(x); free(ys); free(yo); free(Or); free(Oi);
}

/* ---------------------------------------------------------------------------------------------- */
/* Features (contract 2.2 row 1, 20-engine 5.3) against double math                                */
/* ---------------------------------------------------------------------------------------------- */
static void test_features(void) {
  float Xr[257], Xi[257], feat[771], work[3 * 257];
  fill_rand(Xr, 257, 20.0f);
  fill_rand(Xi, 257, 20.0f);
  Xr[3] = Xi[3] = 0.0f;                          /* an exactly silent bin */
  Xr[4] = 1e-6f; Xi[4] = 0.0f;                   /* a nearly silent one */
  es_features(Xr, Xi, feat, work);
  double err = 0;
  for (int k = 0; k < 257; k++) {
    double mag = hypot(Xr[k], Xi[k]);
    double s = pow(mag + 1e-8, -0.7);
    double ref[3] = { Xr[k] * s, Xi[k] * s, 0.1 * log(mag * mag + 1e-8) };
    for (int c = 0; c < 3; c++) err = fmax(err, fabs(feat[3 * k + c] - ref[c]) / fmax(1.0, fabs(ref[c])));
  }
  check_le("features_vs_double", err, 1e-5);
}

/* ---------------------------------------------------------------------------------------------- */
/* Kernels (20-engine 3.3, 6.2) against naive double C                                             */
/* ---------------------------------------------------------------------------------------------- */
static void naive_linear(const float *X, int M, int in, int out, const float *W, const float *b, double *Y) {
  for (int m = 0; m < M; m++)
    for (int o = 0; o < out; o++) {
      double s = b ? b[o] : 0.0;
      for (int i = 0; i < in; i++) s += (double)X[m * in + i] * W[i * out + o];
      Y[m * out + o] = s;
    }
}

static double sigmoid_d(double x) { return 1.0 / (1.0 + exp(-x)); }

/* Naive LSTM over B sequences with the same strides as es_lstm_seq, in double. */
static void naive_lstm_seq(const float *gx, int gx_t, int gx_b, double *hs, int hs_t, int hs_b,
                           int T, int B, int H, const float *whh, int reverse) {
  double *h = calloc((size_t)H, sizeof(double)), *c = calloc((size_t)H, sizeof(double));
  double *g = malloc(sizeof(double) * 4 * (size_t)H), *hn = malloc(sizeof(double) * (size_t)H);
  for (int b = 0; b < B; b++) {
    memset(h, 0, sizeof(double) * (size_t)H);
    memset(c, 0, sizeof(double) * (size_t)H);
    for (int step = 0; step < T; step++) {
      int t = reverse ? T - 1 - step : step;
      for (int k = 0; k < 4 * H; k++) {
        double s = gx[t * gx_t + b * gx_b + k];
        for (int j = 0; j < H; j++) s += h[j] * whh[j * 4 * H + k];
        g[k] = s;
      }
      for (int j = 0; j < H; j++) {
        double i = sigmoid_d(g[j]), f = sigmoid_d(g[H + j]), gg = tanh(g[2 * H + j]), o = sigmoid_d(g[3 * H + j]);
        c[j] = f * c[j] + i * gg;
        hn[j] = o * tanh(c[j]);
      }
      for (int j = 0; j < H; j++) { h[j] = hn[j]; hs[t * hs_t + b * hs_b + j] = h[j]; }
    }
  }
  free(h); free(c); free(g); free(hn);
}

/* Runs es_lstm_seq and the naive version for one stride layout and returns the max abs error. */
static double lstm_case(int T, int B, int H, int in, int gx_t, int gx_b, int hs_t, int hs_b, int hs_rev_off) {
  int rows = T * B;
  float *xin = malloc(sizeof(float) * (size_t)rows * in);
  float *wih = malloc(sizeof(float) * (size_t)in * 4 * H), *wihr = malloc(sizeof(float) * (size_t)in * 4 * H);
  float *whh = malloc(sizeof(float) * (size_t)H * 4 * H), *whhr = malloc(sizeof(float) * (size_t)H * 4 * H);
  float bias[512], biasr[512];
  float *gx = malloc(sizeof(float) * (size_t)rows * 4 * H), *gxr = malloc(sizeof(float) * (size_t)rows * 4 * H);
  float *hs = calloc((size_t)rows * 2 * H, sizeof(float));
  double *ref = calloc((size_t)rows * 2 * H, sizeof(double));
  float *work = malloc(sizeof(float) * (size_t)B * 7 * H);
  fill_rand(xin, rows * in, 1.0f);
  fill_rand(wih, in * 4 * H, 1.0f / sqrtf((float)in));
  fill_rand(wihr, in * 4 * H, 1.0f / sqrtf((float)in));
  fill_rand(whh, H * 4 * H, 1.0f / sqrtf((float)H));
  fill_rand(whhr, H * 4 * H, 1.0f / sqrtf((float)H));
  fill_rand(bias, 4 * H, 0.1f);
  fill_rand(biasr, 4 * H, 0.1f);
  /* Input projections for every row, written into the gx layout (row (t, b) at t*gx_t + b*gx_b). */
  for (int t = 0; t < T; t++)
    for (int b = 0; b < B; b++) {
      const float *xr = xin + (size_t)(t * B + b) * in;
      es_linear(xr, 1, in, 4 * H, wih, bias, gx + (size_t)t * gx_t + (size_t)b * gx_b);
      es_linear(xr, 1, in, 4 * H, wihr, biasr, gxr + (size_t)t * gx_t + (size_t)b * gx_b);
    }
  es_lstm_seq(gx, gx_t, gx_b, hs, hs_t, hs_b, T, B, H, whh, 0, work);
  es_lstm_seq(gxr, gx_t, gx_b, hs + hs_rev_off, hs_t, hs_b, T, B, H, whhr, 1, work);
  naive_lstm_seq(gx, gx_t, gx_b, ref, hs_t, hs_b, T, B, H, whh, 0);
  naive_lstm_seq(gxr, gx_t, gx_b, ref + hs_rev_off, hs_t, hs_b, T, B, H, whhr, 1);
  double err = max_abs_diff_fd(hs, ref, rows * 2 * H);
  free(xin); free(wih); free(wihr); free(whh); free(whhr); free(gx); free(gxr); free(hs); free(ref); free(work);
  return err;
}

static void test_kernels(void) {
  enum { MAXN = 16 * 512 };
  static float X[MAXN], W[128 * 512], Y[MAXN], b[512];
  static double R[MAXN];

  /* es_linear: one row (sgemv) and 16 rows (sgemm), with and without bias */
  fill_rand(X, 48, 1.0f); fill_rand(W, 48 * 64, 1.0f / sqrtf(48.0f)); fill_rand(b, 64, 0.1f);
  es_linear(X, 1, 48, 64, W, b, Y); naive_linear(X, 1, 48, 64, W, b, R);
  check_le("linear_M1_in48_out64", max_abs_diff_fd(Y, R, 64), 1e-5);
  fill_rand(X, 16 * 64, 1.0f); fill_rand(W, 64 * 512, 1.0f / 8.0f); fill_rand(b, 512, 0.1f);
  es_linear(X, 16, 64, 512, W, b, Y); naive_linear(X, 16, 64, 512, W, b, R);
  check_le("linear_M16_in64_out512", max_abs_diff_fd(Y, R, 16 * 512), 1e-5);
  fill_rand(X, 16 * 128, 1.0f); fill_rand(W, 128 * 64, 1.0f / sqrtf(128.0f));
  es_linear(X, 16, 128, 64, W, NULL, Y); naive_linear(X, 16, 128, 64, W, NULL, R);
  check_le("linear_M16_no_bias", max_abs_diff_fd(Y, R, 16 * 64), 1e-5);

  /* residual pattern of 20-engine 6.3: z += bias rows, then z += h * W (accumulate) */
  float z[16 * 64]; double zr[16 * 64];
  fill_rand(z, 16 * 64, 1.0f); fill_rand(b, 64, 0.1f);
  naive_linear(X, 16, 128, 64, W, b, R);
  for (int i = 0; i < 16 * 64; i++) zr[i] = z[i] + R[i];
  es_add_rows(z, 16, 64, b);
  es_linear_acc(X, 16, 128, 64, W, z);
  check_le("linear_acc_residual_M16", max_abs_diff_fd(z, zr, 16 * 64), 1e-5);
  float z1[64]; double z1r[64];
  fill_rand(z1, 64, 1.0f);
  naive_linear(X, 1, 128, 64, W, NULL, R);
  for (int i = 0; i < 64; i++) z1r[i] = z1[i] + R[i];
  es_linear_acc(X, 1, 128, 64, W, z1);
  check_le("linear_acc_M1", max_abs_diff_fd(z1, z1r, 64), 1e-5);

  /* Real-time twins used by the hop (one sgemv per row, never sgemm, so no malloc inside Accelerate):
     same result as naive double, and bit-equal to calling es_linear once per row (M = 1). */
  fill_rand(X, 16 * 128, 1.0f); fill_rand(W, 128 * 512, 1.0f / sqrtf(128.0f)); fill_rand(b, 512, 0.1f);
  es_linear_rt(X, 16, 128, 512, W, b, Y); naive_linear(X, 16, 128, 512, W, b, R);
  check_le("linear_rt_M16_in128_out512", max_abs_diff_fd(Y, R, 16 * 512), 1e-5);
  static float Yrow[MAXN];
  for (int m = 0; m < 16; m++) es_linear(X + 128 * m, 1, 128, 512, W, b, Yrow + 512 * m);
  check_true("linear_rt_equals_rowwise_es_linear_memcmp", memcmp(Y, Yrow, sizeof(float) * 16 * 512) == 0);
  fill_rand(z, 16 * 64, 1.0f);
  for (int i = 0; i < 16 * 64; i++) zr[i] = z[i];
  naive_linear(X, 16, 128, 64, W, NULL, R);
  for (int i = 0; i < 16 * 64; i++) zr[i] += R[i];
  es_linear_acc_rt(X, 16, 128, 64, W, z);
  check_le("linear_acc_rt_M16", max_abs_diff_fd(z, zr, 16 * 64), 1e-5);
  es_linear_rt(X, 16, 64, 256, W, NULL, Y); naive_linear(X, 16, 64, 256, W, NULL, R);
  check_le("linear_rt_M16_no_bias", max_abs_diff_fd(Y, R, 16 * 256), 1e-5);

  /* E-3 fast path (es_linear_xt: cblas_sgemm on X transposed, in column blocks of <= 256) against its
     reference es_linear_rt (one sgemv per row) on every 16-row shape of the hop (20-engine 6.3), plus
     out = 300 (one full block and a 44-column remainder), with bias, without bias and in the accumulate
     form. es_linear_xt is compiled in every build (only the hop's choice is a build flag), so this row
     always tests it. The summation order differs, so the two agree to rounding, not bit for bit; both
     must also match naive double. The scratch xt is 64-byte aligned like the stream's. */
  static const int hop_shapes[5][2] = { { 64, 512 }, { 128, 512 }, { 128, 64 }, { 64, 256 }, { 128, 300 } };
  static _Alignas(64) float Yx[MAXN], Yh[MAXN], xt[128 * 16];
  double worst_ref = 0.0, worst_naive = 0.0;
  int hop_exact = 1;   /* es_linear_hop / es_linear_acc_hop equal the compiled path bit for bit */
  int hop_is_xt = strcmp(es_hop_path, "xt") == 0;
  for (int si = 0; si < 5; si++) {
    int in = hop_shapes[si][0], out = hop_shapes[si][1];
    fill_rand(X, 16 * in, 1.0f); fill_rand(W, in * out, 1.0f / sqrtf((float)in)); fill_rand(b, out, 0.1f);
    for (int with_b = 0; with_b < 2; with_b++) {
      es_linear_rt(X, 16, in, out, W, with_b ? b : NULL, Y);
      es_linear_xt(X, 16, in, out, W, with_b ? b : NULL, Yx, xt);
      es_linear_hop(X, 16, in, out, W, with_b ? b : NULL, Yh, xt);
      hop_exact = hop_exact && memcmp(Yh, hop_is_xt ? Yx : Y, sizeof(float) * 16 * (size_t)out) == 0;
      naive_linear(X, 16, in, out, W, with_b ? b : NULL, R);
      worst_ref = fmax(worst_ref, max_abs_diff_ff(Yx, Y, 16 * out));
      worst_naive = fmax(worst_naive, max_abs_diff_fd(Yx, R, 16 * out));
    }
    for (int i = 0; i < 16 * out; i++) Yh[i] = Yx[i] = Y[i] = urand();   /* accumulate onto the same start */
    for (int i = 0; i < 16 * out; i++) R[i] = Y[i];
    es_linear_acc_rt(X, 16, in, out, W, Y);
    es_linear_acc_xt(X, 16, in, out, W, Yx, xt);
    es_linear_acc_hop(X, 16, in, out, W, Yh, xt);
    hop_exact = hop_exact && memcmp(Yh, hop_is_xt ? Yx : Y, sizeof(float) * 16 * (size_t)out) == 0;
    static double add[MAXN];
    naive_linear(X, 16, in, out, W, NULL, add);
    for (int i = 0; i < 16 * out; i++) R[i] += add[i];
    worst_ref = fmax(worst_ref, max_abs_diff_ff(Yx, Y, 16 * out));
    worst_naive = fmax(worst_naive, max_abs_diff_fd(Yx, R, 16 * out));
  }
  check_le("linear_xt_vs_linear_rt_reference_all_hop_shapes", worst_ref, 1e-5);
  check_le("linear_xt_vs_naive_double_all_hop_shapes", worst_naive, 1e-5);
  printf("INFO hop_path %s (es_linear_hop compiled path; the app ships sgemv)\n", es_hop_path);
  check_true("linear_hop_equals_compiled_path_exactly", hop_exact);

  /* es_layernorm, 16 rows, out of place and in place */
  float lw[64], lb[64], xin[16 * 64], yo[16 * 64];
  for (int i = 0; i < 64; i++) { lw[i] = 1.0f + 0.1f * urand(); lb[i] = 0.1f * urand(); }
  for (int i = 0; i < 16 * 64; i++) xin[i] = 3.0f * urand() + 0.5f;
  es_ln ln = { lw, lb };
  double lref[16 * 64];
  for (int r = 0; r < 16; r++) {
    double mean = 0, var = 0;
    for (int i = 0; i < 64; i++) mean += xin[r * 64 + i];
    mean /= 64;
    for (int i = 0; i < 64; i++) var += (xin[r * 64 + i] - mean) * (xin[r * 64 + i] - mean);
    var /= 64;
    for (int i = 0; i < 64; i++) lref[r * 64 + i] = (xin[r * 64 + i] - mean) / sqrt(var + 1e-5) * lw[i] + lb[i];
  }
  es_layernorm(xin, 16, &ln, yo);
  check_le("layernorm_R16", max_abs_diff_fd(yo, lref, 16 * 64), 1e-5);
  es_layernorm(xin, 16, &ln, xin);
  check_le("layernorm_R16_in_place", max_abs_diff_fd(xin, lref, 16 * 64), 1e-5);

  /* es_lstm_cell: 16 cells of hidden 128 */
  enum { RC = 16, HC = 128 };
  static float g[RC * 4 * HC], h[RC * HC], c[RC * HC], th[RC * HC];
  static double href[RC * HC], cref[RC * HC];
  fill_rand(g, RC * 4 * HC, 3.0f); fill_rand(h, RC * HC, 1.0f); fill_rand(c, RC * HC, 2.0f);
  for (int r = 0; r < RC; r++)
    for (int j = 0; j < HC; j++) {
      const float *gr = g + r * 4 * HC;
      double i = sigmoid_d(gr[j]), f = sigmoid_d(gr[HC + j]), gg = tanh(gr[2 * HC + j]), o = sigmoid_d(gr[3 * HC + j]);
      cref[r * HC + j] = f * c[r * HC + j] + i * gg;
      href[r * HC + j] = o * tanh(cref[r * HC + j]);
    }
  es_lstm_cell(RC, HC, g, h, c, th);
  check_le("lstm_cell_c_R16_H128", max_abs_diff_fd(c, cref, RC * HC), 1e-5);
  check_le("lstm_cell_h_R16_H128", max_abs_diff_fd(h, href, RC * HC), 1e-5);

  /* es_lstm_seq, both directions, with the three stride layouts of 20-engine 6.2 */
  check_le("lstm_seq_stream_band_path_B1_T16_H64",
           lstm_case(16, 1, 64, 64, 256, 0, 128, 0, 64), 1e-5);
  check_le("lstm_seq_token_time_path_B16_T10_H32",
           lstm_case(10, 16, 32, 64, 16 * 128, 128, 16 * 64, 64, 32), 1e-5);
  check_le("lstm_seq_token_band_path_B10_T16_H32",
           lstm_case(16, 10, 32, 64, 128, 16 * 128, 64, 16 * 64, 32), 1e-5);
}

/* ---------------------------------------------------------------------------------------------- */
/* Energy VAD (contract 2.4, 20-engine 7.1)                                                        */
/* ---------------------------------------------------------------------------------------------- */
/* Fill frames [f0, f1] (inclusive, token-frame numbers) of a clip with a 1 kHz tone of amplitude 0.5,
   replacing the noise there. Frame f covers clip samples 512 f - 448 .. 512 f + 63. */
static void put_tone(float *pcm, int n, int f0, int f1) {
  int a = 512 * f0 - 448, b = 512 * f1 + 63;
  if (a < 0) a = 0;
  for (int i = a; i <= b && i < n; i++) pcm[i] = 0.5f * sinf(2.0f * (float)ES_PI * 1000.0f * (float)i / 16000.0f);
}

static int mask_equals(const uint8_t *mask, int F, const int *on, int n_on) {
  for (int f = 0; f < F; f++) {
    int want = 0;
    for (int i = 0; i < n_on; i++) if (on[i] == f) want = 1;
    if (mask[f] != want) return 0;
  }
  return 1;
}

static void test_vad(void) {
  /* Frame counts, and F = 0 reads and writes nothing. */
  int sizes[5] = { 575, 576, 1088, 4672, 5184 }, want_f[5] = { 0, 2, 3, 10, 11 };
  float *pcm = calloc(6000, sizeof(float));
  fill_rand(pcm, 6000, 0.1f);
  for (int s = 0; s < 5; s++) {
    uint8_t mask[16];
    memset(mask, 0xAB, sizeof(mask));
    int F = es_enroll_num_frames(sizes[s]);
    char name[64];
    snprintf(name, sizeof(name), "vad_num_frames_n%d_F%d", sizes[s], want_f[s]);
    check(name, F == want_f[s], F, want_f[s]);
    es_vad(pcm, sizes[s], mask);
    int untouched = 1, written = 1;
    for (int i = F; i < 16; i++) if (mask[i] != 0xAB) untouched = 0;
    for (int i = 0; i < F; i++) if (mask[i] > 1) written = 0;
    snprintf(name, sizeof(name), "vad_writes_exactly_F_bytes_n%d", sizes[s]);
    check_true(name, untouched && written);
  }
  /* F = 0 reads nothing: pcm points at a page that may not be read at all (PROT_NONE), so any read
     would crash the test. The mask must stay untouched too. */
  float *no_read = mmap(NULL, 16384, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
  uint8_t guard[4] = { 0xAB, 0xAB, 0xAB, 0xAB };
  if (no_read != MAP_FAILED) {
    es_vad(no_read, 575, guard);                  /* F = 0 must return before touching pcm */
    munmap(no_read, 16384);
  }
  check_true("vad_F0_reads_nothing", no_read != MAP_FAILED && guard[0] == 0xAB && guard[3] == 0xAB);
  free(pcm);

  /* Bursts in -60 dBFS noise. F = 20. Tone in frames 8-9, 15 and 19 (the last frame).
     Noise frames are near -60 dB, frame 0 near -69 dB (mostly padding), tone frames near -9 dB:
     P10 = sorted[1] (a noise frame), thr = max(P10 + 6, -9 - 40) = about -49 dB, raw = {8, 9, 15, 19},
     dilated mask = {7, 8, 9, 10, 14, 15, 16, 18, 19}. */
  {
    int F = 20, n = 64 + 512 * (F - 1);
    float *x = malloc(sizeof(float) * (size_t)n);
    fill_rand(x, n, 1e-3f * sqrtf(3.0f));         /* uniform noise with RMS 1e-3 = -60 dBFS */
    put_tone(x, n, 8, 9); put_tone(x, n, 15, 15); put_tone(x, n, 19, 19);
    uint8_t mask[20];
    es_vad(x, n, mask);
    int on[] = { 7, 8, 9, 10, 14, 15, 16, 18, 19 };
    check_true("vad_bursts_in_noise_max_minus_40_rule", mask_equals(mask, F, on, 9));
    free(x);
  }
  /* -30 dBFS noise and one tone frame: now P10 + 6 (about -24 dB) sets the threshold. F = 12. */
  {
    int F = 12, n = 64 + 512 * (F - 1);
    float *x = malloc(sizeof(float) * (size_t)n);
    fill_rand(x, n, 0.0316228f * sqrtf(3.0f));
    put_tone(x, n, 5, 5);
    uint8_t mask[12];
    es_vad(x, n, mask);
    int on[] = { 4, 5, 6 };
    check_true("vad_burst_p10_plus_6_rule", mask_equals(mask, F, on, 3));
    free(x);
  }
  /* Digital silence: every E is -100 dB, nothing is strictly above the threshold. */
  {
    int n = 64 + 512 * 9;
    float *x = calloc((size_t)n, sizeof(float));
    uint8_t mask[10];
    es_vad(x, n, mask);
    check_true("vad_silence_all_zero", mask_equals(mask, 10, NULL, 0));
    free(x);
  }
}

/* ---------------------------------------------------------------------------------------------- */
/* json_min (20-engine 4.1): 20 valid and 20 malformed documents, then lookups                     */
/* ---------------------------------------------------------------------------------------------- */
static char *nested(int depth) {               /* "[[[...1...]]]" with `depth` levels */
  char *s = malloc((size_t)depth * 2 + 2);
  int k = 0;
  for (int i = 0; i < depth; i++) s[k++] = '[';
  s[k++] = '1';
  for (int i = 0; i < depth; i++) s[k++] = ']';
  s[k] = '\0';
  return s;
}

static int parses(const char *doc) {
  es_json j;
  es_status st = es_json_parse(&j, doc, (uint32_t)strlen(doc));
  if (st == ES_OK) es_json_free(&j);
  return st == ES_OK;
}

static void test_json(void) {
  char *deep_ok = nested(ES_JSON_MAX_DEPTH), *deep_bad = nested(ES_JSON_MAX_DEPTH + 1);
  char huge[128];
  memset(huge, '9', 100);
  huge[0] = '['; huge[99] = ']'; huge[100] = '\0';          /* a 98-digit number */
  const char *valid[20] = {
    "{}", "[]", "{\"a\":1}", "{\"a\":[1,2,3],\"b\":{\"c\":\"d\"}}",
    "  {\"x\" : -0.5e-3 , \"y\":true,\"z\":false,\"n\":null}  ", "\"str\"", "123",
    "[[[[1]]]]", "{\"esc\":\"a\\\"b\\\\c\\/d\\n\\t\\r\\b\\f\"}", "{\"utf8\":\"caf\xc3\xa9\"}",
    "[0,-0,0.0,1e5,1E+5,1e-05,-12.5e3]", "{\"empty\":\"\",\"arr\":[],\"obj\":{}}", "[true,false,null]",
    "{\"a\":{\"b\":{\"c\":{\"d\":1}}}}", "[1,\"two\",{\"three\":3},[4]]", "{\"k1\":1,\"k2\":2,\"k3\":3}",
    deep_ok, "{\"model_version\":\"1.0.0\"}", "[1e999]", "\n{\n\"a\" :\t[ 1 , 2 ]\n}\n",
  };
  const char *bad[20] = {
    "", "{", "{\"a\":1", "{\"a\":}", "{\"a\" 1}", "{\"a\":1,}", "[1,2,]", "{\"a\":\"\\u0041\"}",
    "{\"a\":\"\\x\"}", "{\"a\":\"line\nbreak\"}", "[01]", "[1.]", "[.5]", "[+1]", "[1e]", "[NaN]",
    deep_bad, huge, "{\"a\":1}{\"b\":2}", "{1:2}",
  };
  int n_valid = 0, n_bad = 0;
  for (int i = 0; i < 20; i++) {
    if (parses(valid[i])) n_valid++;
    else printf("  json valid document %d was rejected: %s\n", i, valid[i]);
    if (!parses(bad[i])) n_bad++;
    else printf("  json malformed document %d was accepted: %s\n", i, bad[i]);
  }
  check("json_20_valid_accepted", n_valid == 20, n_valid, 20);
  check("json_20_malformed_rejected", n_bad == 20, n_bad, 20);
  check_true("json_rejects_tru_and_unterminated", !parses("[tru]") && !parses("\"abc"));
  free(deep_ok); free(deep_bad);

  /* Lookups */
  const char *doc = "{\"format\":\"earshot-testvec\",\"meta\":{\"tolerances\":{\"out\":0.0002,\"vad\":0}},"
                    "\"arrays\":[{\"name\":\"in\",\"shape\":[32000]},{\"name\":\"mask_t\",\"shape\":[187,2]}],"
                    "\"big\":1e999,\"flag\":true}";
  es_json j;
  check_true("json_lookup_doc_parses", es_json_parse(&j, doc, (uint32_t)strlen(doc)) == ES_OK);
  double v = 0;
  int t = es_json_get_path(&j, 0, "meta.tolerances.out");
  check_true("json_path_number", es_json_num(&j, t, &v) == 0 && v == 0.0002);
  check_true("json_str_eq", es_json_str_eq(&j, es_json_get(&j, 0, "format"), "earshot-testvec") &&
                            !es_json_str_eq(&j, es_json_get(&j, 0, "format"), "earshot-test"));
  int arr = es_json_get(&j, 0, "arrays");
  int second = es_json_arr_at(&j, arr, 1);
  int shape = es_json_get(&j, second, "shape");
  check_true("json_arr_at_nested",
             es_json_str_eq(&j, es_json_get(&j, second, "name"), "mask_t") &&
             es_json_num(&j, es_json_arr_at(&j, shape, 1), &v) == 0 && v == 2.0);
  check_true("json_missing_key_and_index", es_json_get(&j, 0, "nope") == -1 && es_json_arr_at(&j, arr, 2) == -1 &&
                                            es_json_get_path(&j, 0, "meta.nope.out") == -1);
  check_true("json_num_overflow_rejected", es_json_num(&j, es_json_get(&j, 0, "big"), &v) == -1);
  check_true("json_num_wrong_type_rejected", es_json_num(&j, es_json_get(&j, 0, "flag"), &v) == -1);
  es_json_free(&j);
}

/* ---------------------------------------------------------------------------------------------- */
/* Container reader (20-engine 4.1) on small hand-made files                                       */
/* ---------------------------------------------------------------------------------------------- */
static void put_u32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

/* Returns a malloc'd copy of s with the first `from` replaced by `to` (s itself if from is not found). */
static char *replace1(const char *s, const char *from, const char *to) {
  const char *at = strstr(s, from);
  size_t n = strlen(s) + strlen(to) + 1;
  char *out = malloc(n);
  if (!at) { memcpy(out, s, strlen(s) + 1); return out; }
  size_t pre = (size_t)(at - s);
  memcpy(out, s, pre);
  strcpy(out + pre, to);
  strcat(out, at + strlen(from));
  return out;
}

/* The three arrays of the default test file: in f32 [4,3], mask u8 [5], cond.person.n i32 [1]. */
static const char *DEFAULT_ARRAYS =
  "[{\"name\":\"in\",\"dtype\":\"f32\",\"shape\":[4,3],\"offset\":0,\"nbytes\":48},"
  "{\"name\":\"mask\",\"dtype\":\"u8\",\"shape\":[5],\"offset\":64,\"nbytes\":5},"
  "{\"name\":\"cond.person.n\",\"dtype\":\"i32\",\"shape\":[1],\"offset\":128,\"nbytes\":4}]";

typedef struct {
  const char *magic, *format, *model_version, *config, *config_hash, *arrays, *sha;
  long long data_bytes;           /* -1 = the real size */
} cfile;

static uint8_t g_data[132];

static void make_data(void) {
  memset(g_data, 0, sizeof(g_data));
  for (int i = 0; i < 12; i++) { float v = 0.5f * (float)i; memcpy(g_data + 4 * i, &v, 4); }
  uint8_t m[5] = { 1, 0, 1, 1, 0 };
  memcpy(g_data + 64, m, 5);
  int32_t k = 42;
  memcpy(g_data + 128, &k, 4);
}

/* Builds a file: preamble, JSON header, zero padding to 64, then g_data. */
static uint8_t *build_file(const cfile *c, size_t *n_out) {
  char chash[65], dsha[65];
  es_sha256_hex(ES_CONFIG_CANON, strlen(ES_CONFIG_CANON), chash);
  es_sha256_hex(g_data, sizeof(g_data), dsha);
  char *hdr = malloc(8192);
  int hl = snprintf(hdr, 8192,
    "{\"format\":\"%s\",\"format_version\":1,\"model\":\"esb16-df\",\"model_version\":\"%s\","
    "\"config\":%s,\"config_hash\":\"%s\",\"arrays\":%s,\"data_bytes\":%lld,\"data_sha256\":\"%s\","
    "\"meta\":{\"tolerances\":{\"vad\":0,\"out\":0.0002}}}",
    c->format ? c->format : "earshot-testvec", c->model_version ? c->model_version : "1.0.0",
    c->config ? c->config : ES_CONFIG_CANON, c->config_hash ? c->config_hash : chash,
    c->arrays ? c->arrays : DEFAULT_ARRAYS, c->data_bytes >= 0 ? c->data_bytes : (long long)sizeof(g_data),
    c->sha ? c->sha : dsha);
  size_t data_start = ((size_t)16 + (size_t)hl + 63) / 64 * 64;
  size_t n = data_start + sizeof(g_data);
  uint8_t *b = calloc(n, 1);
  memcpy(b, c->magic ? c->magic : "EST1", 4);
  put_u32(b + 4, 1);
  put_u32(b + 8, (uint32_t)hl);
  put_u32(b + 12, 0);
  memcpy(b + 16, hdr, (size_t)hl);
  memcpy(b + data_start, g_data, sizeof(g_data));
  free(hdr);
  *n_out = n;
  return b;
}

/* Opens buf and checks the status and that the error line names `field`. */
static void expect_open(const char *name, const uint8_t *buf, size_t n, const char *magic,
                        es_status want, const char *field) {
  es_container ct;
  char err[256];
  es_status st = es_container_open(buf, n, magic, &ct, err, sizeof(err));
  int ok = st == want && (field == NULL || strstr(err, field) != NULL);
  if (!ok) printf("  %s: status %d (want %d), err \"%s\" (want \"%s\")\n", name, st, want, err, field ? field : "");
  check_true(name, ok);
  if (st == ES_OK) es_container_close(&ct);
}

/* Builds a file from c, then checks the open result. */
static void expect_file(const char *name, const cfile *c, es_status want, const char *field) {
  size_t n;
  uint8_t *b = build_file(c, &n);
  expect_open(name, b, n, c->magic ? c->magic : "EST1", want, field);
  free(b);
}

static void test_container(void) {
  make_data();
  cfile base = { 0 };
  base.data_bytes = -1;
  size_t n;
  uint8_t *b = build_file(&base, &n);

  /* The valid file opens, and every array points at the right bytes. */
  es_container ct;
  char err[256];
  es_status st = es_container_open(b, n, "EST1", &ct, err, sizeof(err));
  check_true("container_valid_opens", st == ES_OK);
  if (st == ES_OK) {
    const es_array *a = es_container_find(&ct, "in"), *m = es_container_find(&ct, "mask");
    const es_array *k = es_container_find(&ct, "cond.person.n");
    float f5;
    int32_t kv = 0;
    if (a) memcpy(&f5, (const uint8_t *)a->data + 20, 4);
    if (k) memcpy(&kv, k->data, 4);
    check_true("container_arrays_found", ct.n_arrays == 3 && a && m && k && !es_container_find(&ct, "nope"));
    check_true("container_array_fields",
               a && m && k && a->dtype == ES_DT_F32 && a->ndim == 2 && a->shape[0] == 4 && a->shape[1] == 3 &&
               a->nbytes == 48 && m->dtype == ES_DT_U8 && m->shape[0] == 5 && m->offset == 64 &&
               k->dtype == ES_DT_I32 && (const uint8_t *)m->data == b + ct.data_start + 64);
    check_true("container_array_values", a && k && f5 == 2.5f && kv == 42 && ((const uint8_t *)m->data)[2] == 1);
    check_true("container_hashes", strcmp(ct.config_hash, ES_CONFIG_HASH_HEX) == 0 && strlen(ct.data_hash) == 64);
    es_container_close(&ct);
  }
  cfile w = base;
  w.magic = "ESW1"; w.format = "earshot-weights";
  expect_file("container_valid_weights_file", &w, ES_OK, NULL);

  /* Preamble damage */
  expect_open("container_truncated_preamble", b, 10, "EST1", ES_ERR_FORMAT, "file");
  expect_open("container_truncated_by_1_byte", b, n - 1, "EST1", ES_ERR_FORMAT, "data_bytes");
  expect_open("container_wrong_requested_magic", b, n, "ESW1", ES_ERR_FORMAT, "magic");
  uint8_t *c = malloc(n);
  memcpy(c, b, n); c[0] = 'X';
  expect_open("container_bad_magic", c, n, "EST1", ES_ERR_FORMAT, "magic");
  memcpy(c, b, n); put_u32(c + 4, 2);
  expect_open("container_version_2", c, n, "EST1", ES_ERR_VERSION, "format_version");
  memcpy(c, b, n); put_u32(c + 8, (uint32_t)n);
  expect_open("container_header_len_past_end", c, n, "EST1", ES_ERR_FORMAT, "header_len");
  memcpy(c, b, n); put_u32(c + 12, 1);
  expect_open("container_reserved_nonzero", c, n, "EST1", ES_ERR_FORMAT, "reserved");
  memcpy(c, b, n); c[16] = 'x';
  expect_open("container_header_not_json", c, n, "EST1", ES_ERR_FORMAT, "header");
  memcpy(c, b, n);
  size_t ds = ((size_t)16 + (b[8] | (b[9] << 8) | (b[10] << 16) | ((size_t)b[11] << 24)) + 63) / 64 * 64;
  c[ds + 3] ^= 1;
  expect_open("container_flipped_data_byte", c, n, "EST1", ES_ERR_FORMAT, "data_sha256");
  free(c);

  /* Header fields */
  cfile t = base;
  t.format = "earshot-weights";
  expect_file("container_format_mismatch", &t, ES_ERR_FORMAT, "format");
  t = base; t.model_version = "2.0.0";
  expect_file("container_model_version_2", &t, ES_ERR_VERSION, "model_version");
  t = base; t.model_version = "1.3.12";
  expect_file("container_model_version_1_minor_ok", &t, ES_OK, NULL);

  /* Config */
  char *cfg;
  cfg = replace1(ES_CONFIG_CANON, "\"hop\":64", "\"hop\":32");
  t = base; t.config = cfg;
  expect_file("container_config_wrong_int", &t, ES_ERR_CONFIG, "config.hop");
  free(cfg);
  cfg = replace1(ES_CONFIG_CANON, "\"ln_eps\":1e-05", "\"ln_eps\":1.00001e-05");
  t = base; t.config = cfg;
  expect_file("container_config_wrong_float", &t, ES_ERR_CONFIG, "config.ln_eps");
  free(cfg);
  cfg = replace1(ES_CONFIG_CANON, "32,33]", "32,34]");
  t = base; t.config = cfg;
  expect_file("container_config_wrong_array_element", &t, ES_ERR_CONFIG, "config.band_widths[15]");
  free(cfg);
  cfg = replace1(ES_CONFIG_CANON, ",\"wespk_dim\":256}", "}");
  t = base; t.config = cfg;
  expect_file("container_config_missing_key", &t, ES_ERR_CONFIG, "config.wespk_dim");
  free(cfg);
  cfg = replace1(ES_CONFIG_CANON, "{\"D\":64", "{\"extra\":1,\"D\":64");
  t = base; t.config = cfg;
  expect_file("container_config_extra_key", &t, ES_ERR_CONFIG, "config.extra");
  free(cfg);
  cfg = replace1(ES_CONFIG_CANON, "{\"D\":64", "{\"D\":64,\"D\":64");
  t = base; t.config = cfg;
  expect_file("container_config_duplicate_key", &t, ES_ERR_CONFIG, "config");
  free(cfg);
  cfg = replace1(ES_CONFIG_CANON, "\"blocks\":4", "\"blocks\":4.0");     /* same value, float spelling */
  t = base; t.config = cfg;
  expect_file("container_config_compares_values_not_text", &t, ES_OK, NULL);
  free(cfg);
  t = base; t.config_hash = "0000000000000000000000000000000000000000000000000000000000000000";
  expect_file("container_config_hash_wrong", &t, ES_ERR_CONFIG, "config_hash");

  /* Data size and hash */
  t = base; t.data_bytes = 131;
  expect_file("container_data_bytes_wrong", &t, ES_ERR_FORMAT, "data_bytes");
  t = base; t.sha = "ffff";
  expect_file("container_data_sha_wrong", &t, ES_ERR_FORMAT, "data_sha256");

  /* Arrays */
  char *arr;
  arr = replace1(DEFAULT_ARRAYS, "\"offset\":64", "\"offset\":32");
  t = base; t.arrays = arr;
  expect_file("container_misaligned_offset", &t, ES_ERR_FORMAT, "mask: offset 32");
  free(arr);
  arr = replace1(DEFAULT_ARRAYS, "\"nbytes\":48", "\"nbytes\":44");
  t = base; t.arrays = arr;
  expect_file("container_nbytes_vs_shape", &t, ES_ERR_FORMAT, "in: nbytes");
  free(arr);
  arr = replace1(DEFAULT_ARRAYS, "\"offset\":128", "\"offset\":192");
  t = base; t.arrays = arr;
  expect_file("container_offset_past_data", &t, ES_ERR_FORMAT, "cond.person.n");
  free(arr);
  arr = replace1(DEFAULT_ARRAYS, "\"dtype\":\"u8\"", "\"dtype\":\"f64\"");
  t = base; t.arrays = arr;
  expect_file("container_bad_dtype", &t, ES_ERR_FORMAT, "mask: dtype");
  free(arr);
  arr = replace1(DEFAULT_ARRAYS, "\"offset\":64", "\"offset\":0");
  t = base; t.arrays = arr;
  expect_file("container_offsets_not_increasing", &t, ES_ERR_FORMAT, "previous");
  free(arr);
  arr = replace1(DEFAULT_ARRAYS, "\"name\":\"mask\"", "\"name\":\"in\"");
  t = base; t.arrays = arr;
  expect_file("container_duplicate_name", &t, ES_ERR_FORMAT, "duplicate");
  free(arr);
  arr = replace1(DEFAULT_ARRAYS, "\"shape\":[5]", "\"shape\":[5,1,1,1,1]");
  t = base; t.arrays = arr;
  expect_file("container_too_many_dims", &t, ES_ERR_FORMAT, "shape");
  free(arr);
  free(b);
}

/* ---------------------------------------------------------------------------------------------- */
/* Rings (20-engine 8.1)                                                                           */
/* ---------------------------------------------------------------------------------------------- */
#define RING_TOTAL 50000000ull          /* samples pushed through in16 by the two-thread test */

typedef struct { es_ring *q; int ok; uint64_t done; } ring_job;

/* Counter value of sample i as a float: exact below 2^24, so wrap there. */
static float seq_value(uint64_t i) { return (float)(i & 0xFFFFFF); }

static void *producer_main(void *arg) {
  ring_job *job = arg;
  uint32_t rng = 12345;
  float chunk[1400];
  uint64_t i = 0;
  while (i < RING_TOTAL) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    uint32_t n = 1 + rng % 1366;                         /* like a decimator burst: 1..1366 samples */
    if (n > RING_TOTAL - i) n = (uint32_t)(RING_TOTAL - i);
    while (ring_space(job->q) < n) sched_yield();
    for (uint32_t k = 0; k < n; k++) chunk[k] = seq_value(i + k);
    ring_write(job->q, chunk, n);
    i += n;
  }
  return NULL;
}

static void *consumer_main(void *arg) {
  ring_job *job = arg;
  uint32_t rng = 777;
  float chunk[1400];
  uint64_t i = 0;
  job->ok = 1;
  while (i < RING_TOTAL) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    uint32_t want = 1 + rng % 1366;
    uint32_t have = ring_count(job->q);
    if (have == 0) { sched_yield(); continue; }
    uint32_t n = have < want ? have : want;
    ring_read(job->q, chunk, n);
    for (uint32_t k = 0; k < n; k++)
      if (chunk[k] != seq_value(i + k)) job->ok = 0;
    i += n;
  }
  job->done = i;
  return NULL;
}

/* History ring with two threads: a writer at full speed (chunks of 1..1,600 like render) and a reader
   that keeps re-reading from the oldest held sample, the riskiest place to read. Sample i holds the
   value i % HIST_PERIOD, so a slot the writer overwrote during a copy shows a wrong value. */
#define HIST_TOTAL  30000000ull
#define HIST_PERIOD (1u << 20)

typedef struct { es_hist *h; const float *src; _Atomic int done; } hist_job;

static void *hist_writer_main(void *arg) {
  hist_job *job = arg;
  uint32_t rng = 4242;
  uint64_t i = 0;
  while (i < HIST_TOTAL) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    uint32_t n = 1 + rng % 1600;
    hist_write(job->h, job->src + (i % HIST_PERIOD), n);   /* src[k] = k % HIST_PERIOD, k < PERIOD + 1600 */
    i += n;
  }
  atomic_store(&job->done, 1);
  return NULL;
}

static void test_hist_two_threads(void) {
  enum { HS = 65536 };
  uint32_t keep = HS - ES_HIST_GUARD;
  float *src = malloc(sizeof(float) * (HIST_PERIOD + 1600));
  for (uint32_t k = 0; k < HIST_PERIOD + 1600; k++) src[k] = (float)(k % HIST_PERIOD);
  float *hb = calloc(HS, sizeof(float)), *dst = malloc(sizeof(float) * keep);
  es_hist h;
  hist_init(&h, hb, HS);
  hist_job job = { &h, src, 0 };
  pthread_t tw;
  pthread_create(&tw, NULL, hist_writer_main, &job);
  int ok = 1;
  long reads = 0, good = 0, thrown = 0, good_newest = 0;
  while (!atomic_load(&job.done)) {
    /* Even reads start at the oldest held sample and copy as much as the ring allows (often thrown
       away: any write published during the copy may overlap it). Odd reads copy the newest 4,096. */
    uint64_t cursor = 0;                                    /* 0 = the oldest held sample */
    int max = (int)keep;
    if (reads % 2 == 1) {
      uint64_t w = hist_position(&h);
      cursor = w > 4096 ? w - 4096 : 0;
      max = 4096;
    }
    int got = hist_read(&h, &cursor, dst, max);
    reads++;
    if (got == 0) { thrown++; continue; }                   /* overrun detected (or nothing new) */
    uint64_t start = cursor - (uint64_t)got;
    for (int k = 0; k < got; k++)
      if (dst[k] != (float)((start + (uint64_t)k) % HIST_PERIOD)) { ok = 0; break; }
    good++;
    if (max == 4096) good_newest++;
  }
  pthread_join(tw, NULL);
  printf("  hist two threads: %ld reads, %ld exact copies (%ld of the newest 4,096), %ld returned 0 (overrun)\n",
         reads, good, good_newest, thrown);
  check_true("hist_two_threads_every_returned_copy_exact", ok && good > 0);
  free(src); free(hb); free(dst);
}

static void test_rings(void) {
  /* Single thread: wrap-around, counts, skip. */
  float buf[8], out[8];
  es_ring q;
  ring_init(&q, buf, 8);
  float a[6] = { 1, 2, 3, 4, 5, 6 };
  ring_write(&q, a, 6);
  ring_read(&q, out, 4);                                  /* r = 4, w = 6 */
  ring_write(&q, a, 5);                                   /* wraps: w = 11 */
  int ok = ring_count(&q) == 7 && ring_space(&q) == 1;
  ring_read(&q, out, 7);
  float want[7] = { 5, 6, 1, 2, 3, 4, 5 };
  ok = ok && memcmp(out, want, sizeof(want)) == 0;
  ring_write(&q, a, 3);
  ring_skip(&q, 2);
  ring_read(&q, out, 1);
  ok = ok && out[0] == 3 && ring_count(&q) == 0 && ring_space(&q) == 8 && ring_total_read(&q) == 14;
  check_true("ring_wraparound_count_space_skip", ok);

  uint16_t b16[4], o16[4];
  es_ring16 q16;
  ring16_init(&q16, b16, 4);
  for (uint16_t v = 1; v <= 3; v++) ring16_write1(&q16, v);
  uint32_t got = ring16_read(&q16, o16, 2);
  for (uint16_t v = 4; v <= 6; v++) ring16_write1(&q16, v);   /* wraps */
  got += ring16_read(&q16, o16, 4);
  check_true("ring16_wraparound", got == 6 && o16[0] == 3 && o16[1] == 4 && o16[3] == 6 && ring16_space(&q16) == 4);

  /* Two threads, 50 M samples of a counter, random chunk sizes on both sides, size 4,096 (in16). */
  float *big = calloc(4096, sizeof(float));
  es_ring in16;
  ring_init(&in16, big, 4096);
  ring_job job = { &in16, 0, 0 };
  pthread_t tp, tc;
  pthread_create(&tc, NULL, consumer_main, &job);
  pthread_create(&tp, NULL, producer_main, &job);
  pthread_join(tp, NULL);
  pthread_join(tc, NULL);
  check_true("ring_two_threads_50M_sequence_exact", job.ok && job.done == RING_TOTAL);
  free(big);

  /* History ring: cursors, the oldest-held rule and the skip count (single thread). */
  enum { HS = 65536 };
  float *hb = calloc(HS, sizeof(float)), *dst = malloc(sizeof(float) * HS), *chunk = malloc(sizeof(float) * 4096);
  es_hist h;
  hist_init(&h, hb, HS);
  uint64_t written = 0;
  for (int k = 0; k < 10000 / 1000; k++) {               /* 10,000 samples */
    for (int i = 0; i < 1000; i++) chunk[i] = seq_value(written + (uint64_t)i);
    hist_write(&h, chunk, 1000);
    written += 1000;
  }
  uint64_t cursor = 0;
  int n1 = hist_read(&h, &cursor, dst, HS);
  ok = n1 == 10000 && cursor == 10000 && dst[0] == 0.0f && dst[9999] == 9999.0f;
  int n0 = hist_read(&h, &cursor, dst, HS);               /* nothing new yet */
  ok = ok && n0 == 0 && cursor == 10000 && hist_position(&h) == 10000;
  check_true("hist_read_from_start_then_nothing", ok);
  while (written < 110000) {                             /* 100,000 more: cursor 10,000 is no longer held */
    for (int i = 0; i < 4000; i++) chunk[i] = seq_value(written + (uint64_t)i);
    hist_write(&h, chunk, 4000);
    written += 4000;
  }
  uint64_t old = cursor;
  int n2 = hist_read(&h, &cursor, dst, HS);
  uint64_t oldest = written - (HS - ES_HIST_GUARD);      /* 110,000 - 49,152 = 60,848 */
  uint64_t skipped = cursor - old - (uint64_t)n2;
  ok = n2 == (int)(written - oldest) && dst[0] == seq_value(oldest) && cursor == written &&
       skipped == oldest - old;
  check_true("hist_old_cursor_starts_at_oldest_and_counts_skip", ok);
  int n3 = hist_read_last(&h, dst, 48000);
  check_true("hist_read_last_48000", n3 == 48000 && dst[47999] == seq_value(written - 1) &&
                                     dst[0] == seq_value(written - 48000));
  free(hb); free(dst); free(chunk);

  test_hist_two_threads();
}

/* ---------------------------------------------------------------------------------------------- */
/* Resampler (20-engine 9): response from impulses, block-size independence, need, prefill         */
/* ---------------------------------------------------------------------------------------------- */
/* Rebuild the prototype h at the high rate (L * I/O or L * 16 kHz) by running the resampler on unit
   impulses. With an impulse at input d, output m equals L * h[m M - d L] (when that index is inside
   the filter), so impulses at d = 0 .. M-1 visit every tap once (L and M share no factor).
   Returns h in a malloc'd array of Q * L doubles. */
static double *impulse_prototype(const es_rs_table *t, int *len_out) {
  int len = t->Q * t->L;
  double *h = calloc((size_t)len, sizeof(double));
  int n_in = t->M + t->Q + 2;                         /* past the last output an impulse can reach */
  int max_out = (int)((int64_t)n_in * t->L / t->M) + 2;
  float *x = malloc(sizeof(float) * (size_t)n_in), *y = malloc(sizeof(float) * (size_t)max_out);
  float *hist = malloc(sizeof(float) * (size_t)es_rs_hist_len(t, n_in));
  for (int d = 0; d < t->M; d++) {
    memset(x, 0, sizeof(float) * (size_t)n_in);
    x[d] = 1.0f;
    es_rs r;
    es_rs_init(&r, t, hist);
    int k = es_rs_process(&r, x, n_in, y, max_out);
    for (int m = 0; m < k; m++) {
      int64_t n0 = (int64_t)m * t->M;                  /* high-rate position of output m */
      int64_t base = n0 / t->L, p = n0 % t->L, j = base - d;
      if (j >= 0 && j < t->Q) h[p + j * t->L] = (double)y[m] / t->L;
    }
  }
  free(x); free(y); free(hist);
  *len_out = len;
  return h;
}

/* Response of h at frequency f (Hz) for sample rate fs: H and, if hn != NULL, the response of n h[n]. */
static void dtft_at(const double *h, int len, double fs, double f, double *re, double *im, double *nre, double *nim) {
  double w = -2.0 * ES_PI * f / fs, cr = cos(w), ci = sin(w);
  double zr = 1.0, zi = 0.0, sr = 0, si = 0, tr = 0, ti = 0;
  for (int n = 0; n < len; n++) {
    sr += h[n] * zr; si += h[n] * zi;
    tr += n * h[n] * zr; ti += n * h[n] * zi;
    double nzr = zr * cr - zi * ci;                    /* z *= e^{-i w}, a rotation */
    zi = zr * ci + zi * cr;
    zr = nzr;
  }
  *re = sr; *im = si;
  if (nre) { *nre = tr; *nim = ti; }
}

static double db_at(const double *h, int len, double fs, double f) {
  double re, im;
  dtft_at(h, len, fs, f, &re, &im, NULL, NULL);
  return 20.0 * log10(hypot(re, im));
}

/* Highest stopband level from 8 kHz up to fs / 2: a fine direct scan to 24 kHz (where the first
   sidelobes are), plus a 2^17-point FFT for the rest of the band (the 7.056 MHz prototype). */
static double stopband_db(const double *h, int len, double fs) {
  double worst = -400;
  for (double f = 8000.0; f <= 24000.0; f += (len < 1000 ? 2.0 : 10.0)) worst = fmax(worst, db_at(h, len, fs, f));
  if (fs > 48000.0) {
    int log2n = 17, N = 1 << log2n;
    float *buf = calloc((size_t)N, sizeof(float)), *re = malloc(sizeof(float) * N / 2), *im = malloc(sizeof(float) * N / 2);
    for (int n = 0; n < len; n++) buf[n] = (float)h[n];
    FFTSetup s = vDSP_create_fftsetup(log2n, kFFTRadix2);
    DSPSplitComplex sp = { re, im };
    vDSP_ctoz((const DSPComplex *)buf, 2, &sp, 1, (vDSP_Length)(N / 2));
    vDSP_fft_zrip(s, &sp, 1, log2n, kFFTDirection_Forward);
    for (int k = 1; k < N / 2; k++) {
      double f = k * fs / N;
      if (f < 24000.0) continue;
      double mag = 0.5 * hypot(re[k], im[k]);          /* vDSP forward is 2x the DFT */
      worst = fmax(worst, 20.0 * log10(mag + 1e-300));
    }
    vDSP_destroy_fftsetup(s);
    free(buf); free(re); free(im);
  }
  return worst;
}

typedef struct { double ripple_lo, ripple_hi, stop, gd_mean_ms, f3db; } rs_figs;

static rs_figs measure(const es_rs_table *t, double fs) {
  int len;
  double *h = impulse_prototype(t, &len);
  rs_figs r = { 1e9, -1e9, 0, 0, 0 };
  for (double f = 0.0; f <= 6500.0; f += 5.0) {
    double d = db_at(h, len, fs, f);
    r.ripple_lo = fmin(r.ripple_lo, d);
    r.ripple_hi = fmax(r.ripple_hi, d);
  }
  r.stop = stopband_db(h, len, fs);
  double sum = 0;
  int cnt = 0;
  for (double f = 1000.0; f <= 4000.0 + 1e-9; f += 10.0) {     /* group delay = Re(DTFT(n h) / DTFT(h)) */
    double re, im, nre, nim;
    dtft_at(h, len, fs, f, &re, &im, &nre, &nim);
    sum += (nre * re + nim * im) / (re * re + im * im);
    cnt++;
  }
  r.gd_mean_ms = sum / cnt / fs * 1000.0;
  double prev_f = 6500.0, prev_d = db_at(h, len, fs, 6500.0);  /* -3 dB point, 0.5 Hz scan */
  for (double f = 6500.5; f <= 8000.0; f += 0.5) {
    double d = db_at(h, len, fs, f);
    if (d < -3.0) { r.f3db = prev_f + (f - prev_f) * (-3.0 - prev_d) / (d - prev_d); break; }
    prev_f = f; prev_d = d;
  }
  free(h);
  return r;
}

/* Run a resampler over x in random block sizes (1..max_in) or one block (max_in = n), collect output. */
static int run_blocks(const es_rs_table *t, const float *x, int n, int max_in, int random_blocks, float *y, int max_y) {
  float *hist = malloc(sizeof(float) * (size_t)es_rs_hist_len(t, max_in));
  es_rs r;
  es_rs_init(&r, t, hist);
  uint32_t rng = 99;
  int pos = 0, total = 0;
  while (pos < n) {
    int blk = max_in;
    if (random_blocks) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; blk = 1 + (int)(rng % (uint32_t)max_in); }
    if (blk > n - pos) blk = n - pos;
    total += es_rs_process(&r, x + pos, blk, y + total, max_y - total);
    pos += blk;
  }
  free(hist);
  return total;
}

static void test_resampler(void) {
  /* 1. Response of every table, measured from impulses through es_rs_process (20-engine 9.1). */
  const char *fam_name[2] = { "48k", "44k" };
  const double fs_proto[2] = { 48000.0, 7056000.0 };
  const double want_lo[2] = { -0.0031, -0.0031 }, want_hi[2] = { 0.0042, 0.0043 };
  const double want_gd[2] = { 0.1288, 0.1494 }, want_delay[2] = { 0.2575, 0.2987 };
  for (int fam = 0; fam < 2; fam++) {
    double gd_sum = 0;
    for (int dir = 0; dir < 2; dir++) {
      const es_rs_table *t = dir == 0 ? &es_rs_decim_tables[fam] : &es_rs_interp_tables[fam];
      rs_figs f = measure(t, fs_proto[fam]);
      char name[96];
      const char *dn = dir == 0 ? "decim" : "interp";
      printf("  resampler %s %s: passband %+.4f to %+.4f dB, stopband %.2f dB, mean group delay %.5f ms, "
             "-3 dB at %.1f Hz\n", fam_name[fam], dn, f.ripple_lo, f.ripple_hi, f.stop, f.gd_mean_ms, f.f3db);
      snprintf(name, sizeof(name), "resampler_%s_%s_ripple_low", fam_name[fam], dn);
      check_le(name, fabs(f.ripple_lo - want_lo[fam]), 0.01);
      snprintf(name, sizeof(name), "resampler_%s_%s_ripple_high", fam_name[fam], dn);
      check_le(name, fabs(f.ripple_hi - want_hi[fam]), 0.01);
      snprintf(name, sizeof(name), "resampler_%s_%s_stopband_60dB", fam_name[fam], dn);
      check_le(name, fabs(f.stop - (-60.0)), 1.0);
      snprintf(name, sizeof(name), "resampler_%s_%s_mean_group_delay", fam_name[fam], dn);
      check_le(name, fabs(f.gd_mean_ms - want_gd[fam]), 0.01);
      if (fam == 0) {
        snprintf(name, sizeof(name), "resampler_48k_%s_minus3dB_7385Hz", dn);
        check_le(name, fabs(f.f3db - 7385.0), 5.0);
      }
      gd_sum += f.gd_mean_ms;
    }
    char name[96];
    snprintf(name, sizeof(name), "resampler_%s_ES_RS_DELAY_MS_vs_spec", fam_name[fam]);
    check_le(name, fabs(ES_RS_DELAY_MS[fam] - want_delay[fam]), 0.01);
    snprintf(name, sizeof(name), "resampler_%s_ES_RS_DELAY_MS_vs_measured", fam_name[fam]);
    check_le(name, fabs(ES_RS_DELAY_MS[fam] - gd_sum), 0.001);
  }

  /* 2. Random block sizes give bit-identical output to one big block (both directions, both rates). */
  enum { NX = 48000 };
  float *x = malloc(sizeof(float) * NX), *y1 = malloc(sizeof(float) * 2 * NX), *y2 = malloc(sizeof(float) * 2 * NX);
  fill_rand(x, NX, 0.5f);
  for (int fam = 0; fam < 2; fam++)
    for (int dir = 0; dir < 2; dir++) {
      const es_rs_table *t = dir == 0 ? &es_rs_decim_tables[fam] : &es_rs_interp_tables[fam];
      int max_in = dir == 0 ? ES_RS_MAX_DECIM_IN : ES_RS_MAX_INTERP_IN;
      int n = dir == 0 ? NX : NX / 3;
      int k1 = run_blocks(t, x, n, n, 0, y1, 2 * NX);
      int k2 = run_blocks(t, x, n, max_in, 1, y2, 2 * NX);
      char name[96];
      snprintf(name, sizeof(name), "resampler_%s_%s_random_blocks_bit_identical", fam_name[fam], dir == 0 ? "decim" : "interp");
      check_true(name, k1 == k2 && k1 > 0 && memcmp(y1, y2, sizeof(float) * (size_t)k1) == 0);
    }

  free(x); free(y1); free(y2);

  /* 3. The interpolator driven the way render drives it (20-engine 8.3): feed exactly es_rs_need(n)
     new inputs and ask for max_out = n. Every call must return exactly n outputs, the count-only twin
     must agree, and the samples must be bit-identical to one big block over the same input.
     Every fourth call asks for only 1 to 8 outputs, which often needs 0 new inputs. Such a call starts
     with base = consumed - 1, so its first output needs Q old samples (the case that once read one
     float before hist). */
  enum { NIN = 600000, NCALL = 600, NOUT = NCALL * 4096 };
  float *xin = malloc(sizeof(float) * NIN);
  float *yneed = malloc(sizeof(float) * NOUT), *yone = malloc(sizeof(float) * NOUT);
  fill_rand(xin, NIN, 0.5f);
  for (int fam = 0; fam < 2; fam++) {
    const es_rs_table *t = &es_rs_interp_tables[fam];
    float *hist = malloc(sizeof(float) * (size_t)es_rs_hist_len(t, ES_RS_MAX_INTERP_IN));
    es_rs r, rc;
    es_rs_init(&r, t, hist);
    es_rs_init(&rc, t, NULL);
    int ok = 1, pos = 0, total = 0, zero_need_calls = 0;
    uint32_t rng = 5;
    for (int call = 0; call < NCALL; call++) {
      rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
      int n = (call % 4 == 0) ? 1 + (int)(rng % 8u) : 1 + (int)(rng % 4096u);
      int need = es_rs_need(&r, n);
      if (need > ES_RS_MAX_INTERP_IN || pos + need > NIN) { ok = 0; break; }
      if (need == 0) zero_need_calls++;
      int got = es_rs_process(&r, xin + pos, need, yneed + total, n);
      int cnt = es_rs_step_count(&rc, need, n);
      if (got != n || cnt != n || rc.base != r.base || rc.p != r.p) ok = 0;
      pos += need;
      total += got;
    }
    char name[80];
    snprintf(name, sizeof(name), "resampler_%s_need_gives_exact_outputs", fam_name[fam]);
    check_true(name, ok && zero_need_calls > 0);

    /* The same pos inputs in one block (max_out large) give the reference samples. */
    int k1 = run_blocks(t, xin, pos, pos, 0, yone, NOUT);
    double worst = 0;
    for (int i = 0; i < total && i < k1; i++) worst = fmax(worst, fabs((double)yneed[i] - (double)yone[i]));
    int same = ok && k1 >= total && memcmp(yneed, yone, sizeof(float) * (size_t)total) == 0;
    snprintf(name, sizeof(name), "resampler_%s_need_driven_bit_identical_to_one_block", fam_name[fam]);
    check(name, same, worst, 0);
    free(hist);
  }
  free(xin); free(yneed); free(yone);

  /* 4. Prefill P_min equals the 20-engine 9.4 table. */
  const int sizes[13] = { 16, 32, 48, 64, 96, 128, 192, 256, 480, 512, 1024, 1104, 4096 };
  const int p48[13] = { 64, 64, 64, 64, 64, 86, 64, 128, 192, 214, 384, 416, 1408 };
  const int p44[13] = { 69, 75, 80, 86, 98, 110, 132, 156, 237, 249, 435, 463, 1549 };
  int bad = 0;
  for (int i = 0; i < 13; i++) {
    int a = es_rs_prefill_min(48000, sizes[i]), b = es_rs_prefill_min(44100, sizes[i]);
    if (a != p48[i]) { printf("  P_min 48k %d frames: %d, table %d\n", sizes[i], a, p48[i]); bad++; }
    if (b != p44[i]) { printf("  P_min 44.1k %d frames: %d, table %d\n", sizes[i], b, p44[i]); bad++; }
  }
  check("prefill_pmin_matches_table_26_entries", bad == 0, bad, 0);
  check_true("prefill_bad_rate_or_size", es_rs_prefill_min(32000, 64) == -1 && es_rs_prefill_min(48000, 0) == -1 &&
                                         es_rs_family(96000) == -1);
}

/* ---------------------------------------------------------------------------------------------- */
/* Ramps, limiter, soft clip (20-engine 11), defaults, status strings, atomics                     */
/* ---------------------------------------------------------------------------------------------- */
static void test_ramps(void) {
  enum { NB = 19400 };
  float *buf = malloc(sizeof(float) * NB);
  es_ramp r;

  /* Linear 5 ms at 48 kHz = 240 samples: sample 240 (index 239) is the target, sample 239 is not. */
  es_ramp_init(&r, 48000.0f, 1.0f);
  es_ramp_to(&r, 0.0f, 5.0f, 0);
  for (int i = 0; i < NB; i++) buf[i] = 1.0f;
  es_ramp_process(&r, buf, 300);
  check_le("ramp_linear_5ms_ends_at_240", fabs(buf[239]), 1e-6);
  check("ramp_linear_5ms_not_done_at_239", fabs(buf[238]) > 1e-6, fabs(buf[238]), 1e-6);
  check_le("ramp_linear_first_step", fabs(buf[0] - (1.0 - 1.0 / 240.0)), 1e-6);
  double dev = 0;                                  /* every sample vs the ideal 1 - (i + 1) / 240 */
  for (int i = 0; i < 300; i++) dev = fmax(dev, fabs(buf[i] - fmax(0.0, 1.0 - (i + 1) / 240.0)));
  check_le("ramp_linear_5ms_tracks_ideal", dev, 1e-6);

  /* Raised cosine 400 ms at 48 kHz = 19,200 samples, rising 0 -> 1. The phase is a float that adds
     step = 1 / 19,200 per sample (20-engine 11, the contract's es_ramp fields are float), so rounding
     makes it reach 1 about 2 samples early (measured: after 19,198 samples), and mid-ramp the gain is
     at most about 1.2e-4 from the ideal curve. Both are far below hearing; the checks pin them. */
  es_ramp_init(&r, 48000.0f, 0.0f);
  es_ramp_to(&r, 1.0f, 400.0f, 1);
  int finish = -1;
  dev = 0;
  for (int i = 0; i < NB; i++) {
    buf[i] = 1.0f;
    es_ramp_process(&r, &buf[i], 1);
    if (finish < 0 && r.phase >= 1.0f) finish = i + 1;        /* samples until the ramp is done */
    double ideal = 0.5 - 0.5 * cos(ES_PI * fmin(1.0, (i + 1) / 19200.0));
    dev = fmax(dev, fabs(buf[i] - ideal));
  }
  check_le("ramp_raised_cosine_400ms_finishes_within_4_of_19200", fabs(finish - 19200.0), 4.0);
  check_le("ramp_raised_cosine_400ms_tracks_ideal", dev, 2e-4);
  check_le("ramp_raised_cosine_value_at_19200", fabs(buf[19199] - 1.0), 1e-6);
  check("ramp_raised_cosine_not_done_200_early", fabs(buf[18999] - 1.0) > 1e-6, fabs(buf[18999] - 1.0), 1e-6);
  check_le("ramp_raised_cosine_finished", fabs(r.gain - 1.0) + fabs(r.phase - 1.0), 0.0);

  /* Unmute at gain 0.5 during a mute ramp continues from 0.5 (the start field), no jump. */
  es_ramp_init(&r, 48000.0f, 1.0f);
  es_ramp_to(&r, 0.0f, 5.0f, 0);
  for (int i = 0; i < 120; i++) buf[i] = 1.0f;
  es_ramp_process(&r, buf, 120);                   /* gain is now 0.5 */
  float before = r.gain;
  es_ramp_to(&r, 1.0f, 400.0f, 1);
  buf[0] = 1.0f;
  es_ramp_process(&r, buf, 1);
  check_le("ramp_unmute_at_half_starts_at_0.5", fabs(before - 0.5), 1e-6);
  check_le("ramp_unmute_no_jump", fabs(buf[0] - before) + fabs(r.start - before), 1e-6);

  /* A falling raised cosine never rises, and lands exactly on 0. */
  es_ramp_init(&r, 48000.0f, 1.0f);
  es_ramp_to(&r, 0.0f, 400.0f, 1);
  for (int i = 0; i < NB; i++) buf[i] = 1.0f;
  es_ramp_process(&r, buf, NB);
  int mono = 1;
  for (int i = 1; i < NB; i++) if (buf[i] > buf[i - 1]) mono = 0;
  check_true("ramp_falling_raised_cosine_monotonic", mono && buf[NB - 1] == 0.0f);

  /* ms <= 0 jumps at once. */
  es_ramp_to(&r, 0.25f, 0.0f, 0);
  buf[0] = 1.0f;
  es_ramp_process(&r, buf, 1);
  check_true("ramp_zero_ms_jumps", buf[0] == 0.25f && r.gain == 0.25f);
  free(buf);
}

static void test_limiter_softclip(void) {
  es_limiter l;
  es_limiter_init(&l, 48000.0f, -3.0f, 50.0f);
  check_le("limiter_ceiling_minus3dB", fabs(l.ceiling - 0.7079458), 1e-6);
  check_le("limiter_release_coef_48k", fabs(l.rel_coef - 0.9995834), 1e-6);

  /* 10 M random samples with magnitudes up to 1,000 never leave the ceiling. */
  float chunk[4096];
  double worst = 0;
  for (int c = 0; c < 10000000 / 4096 + 1; c++) {
    for (int i = 0; i < 4096; i++) chunk[i] = urand() * powf(10.0f, 3.0f * fabsf(urand()));
    es_limiter_process(&l, chunk, 4096);
    for (int i = 0; i < 4096; i++) worst = fmax(worst, fabs(chunk[i]));
  }
  check_le("limiter_10M_samples_never_above_ceiling", worst - l.ceiling, 1e-6);

  /* Release: from a gain of 0.1, gain = 1 - 0.9 e^-1 after 50 ms = 2,400 samples. */
  es_limiter_init(&l, 48000.0f, -3.0f, 50.0f);
  float x = l.ceiling / 0.1f;
  es_limiter_process(&l, &x, 1);
  float g0 = l.gain;
  double target = 1.0 - 0.9 * exp(-1.0);
  int cross = -1;
  double g2400 = 0;
  for (int k = 1; k <= 5000; k++) {
    float z = 0.0f;
    es_limiter_process(&l, &z, 1);
    if (cross < 0 && l.gain >= target) cross = k;
    if (k == 2400) g2400 = l.gain;
  }
  check_le("limiter_attack_to_0.1", fabs(g0 - 0.1), 1e-6);
  check_le("limiter_release_reaches_1-0.9/e_at_2400", fabs(cross - 2400.0), 1.0);
  /* one sample of release moves the gain by about 0.9 e^-1 / 2400 = 1.4e-4 */
  check_le("limiter_release_gain_at_2400", fabs(g2400 - target), 1.4e-4);

  /* Soft clip */
  const float k = 0.8912509f;
  float v[8];
  int ident = 1;
  for (int i = 0; i < 1000; i++) {
    float a = 0.89f * urand(), b = a;
    es_softclip(&b, 1);
    if (b != a) ident = 0;
  }
  check_true("softclip_identity_below_knee", ident);
  v[0] = k + 1e-4f;
  es_softclip(v, 1);
  check_le("softclip_continuous_at_knee", fabs(v[0] - (k + 1e-4f)), 1e-6);
  v[0] = k + 1e-3f; v[1] = k + 2e-3f;
  es_softclip(v, 2);
  check_le("softclip_slope_1_at_knee", fabs((v[1] - v[0]) / 1e-3 - 1.0), 1e-3);
  float big[6] = { 1000.0f, -1000.0f, 10.0f, -1.0001f, INFINITY, -INFINITY };
  es_softclip(big, 6);
  double bmax = 0;
  for (int i = 0; i < 6; i++) bmax = fmax(bmax, fabs(big[i]));
  check_le("softclip_bounded_by_1", bmax, 1.0);
  check_true("softclip_keeps_sign", big[1] < 0 && big[5] < 0 && big[0] > 0);
}

/* ---------------------------------------------------------------------------------------------- */
/* Tensor list and synthetic models (contract 3.2, 20-engine 4.4)                                  */
/* ---------------------------------------------------------------------------------------------- */
static void test_model_synthetic(void) {
  /* The list: 270 tensors, unique names, 2,570,500 floats (contract 3.2). */
  static es_tensor list[ES_N_TENSORS];
  es_model *m = NULL, *m2 = NULL, *ap = NULL;
  check_true("synthetic_bad_arg", es_model_synthetic(1, 0, NULL) == ES_ERR_ARG);
  int st = es_model_synthetic(1, 0, &m);
  check_true("synthetic_model_builds", st == ES_OK && m != NULL);
  if (st != ES_OK) return;
  int n = es_model_tensors(m, list);
  long floats = 0;
  int unique = 1;
  for (int i = 0; i < n; i++) {
    floats += (long)list[i].d0 * (list[i].d1 > 0 ? list[i].d1 : 1);
    for (int j = 0; j < i; j++) if (strcmp(list[i].name, list[j].name) == 0) unique = 0;
  }
  check("tensor_list_270_unique_names", n == ES_N_TENSORS && unique, n, ES_N_TENSORS);
  check("tensor_list_2570500_floats", floats == ES_N_WEIGHT_FLOATS, (double)floats, ES_N_WEIGHT_FLOATS);
  check_true("tensor_list_first_and_last", strcmp(list[0].name, "enc.lin.0.weight") == 0 && list[0].d0 == 24 &&
                                           strcmp(list[n - 1].name, "tok.g.bias") == 0 && list[n - 1].d0 == 128);

  /* Arena layout: every field points at a 64-byte aligned tensor inside the arena, in list order,
     and the arena is 10,282,112 bytes (the data_bytes of an export with the same layout). */
  int layout_ok = 1;
  const char *base = m->arena, *prev = NULL;
  for (int i = 0; i < n; i++) {
    const char *p = (const char *)*list[i].dst;
    size_t bytes = sizeof(float) * (size_t)list[i].d0 * (size_t)(list[i].d1 > 0 ? list[i].d1 : 1);
    if (!p || (size_t)(p - base) % 64 != 0 || p < base || p + bytes > base + m->arena_bytes || p <= prev)
      layout_ok = 0;
    prev = p;
  }
  check_true("synthetic_arena_layout_aligned_in_order", layout_ok);
  check("synthetic_arena_bytes_10282112", m->arena_bytes == 10282112, (double)m->arena_bytes, 10282112);
  check_true("synthetic_band_tables", m->band_start[15] == 224 && m->band_width[15] == 33 &&
                                      m->band_taps[11] == 3 && m->band_taps[12] == 1 && m->coef_off[15] == 1026 - 66);

  /* Value ranges of 20-engine 4.4. */
  int range_ok = 1;
  for (int i = 0; i < n; i++) {
    const float *w = *list[i].dst;
    int count = list[i].d0 * (list[i].d1 > 0 ? list[i].d1 : 1);
    float lim = list[i].init == ES_INIT_WEIGHT ? 1.0f / sqrtf((float)list[i].d0) : 0.1f;
    for (int j = 0; j < count; j++) {
      float v = list[i].init == ES_INIT_LN_WEIGHT ? w[j] - 1.0f : w[j];
      if (!(fabsf(v) <= lim * 1.000001f)) range_ok = 0;
    }
  }
  check_true("synthetic_value_ranges", range_ok);

  /* Same seed, same model; another seed, another model. */
  es_model_synthetic(1, 0, &m2);
  int same = m2 && memcmp(m->arena, m2->arena, m->arena_bytes) == 0 &&
             strcmp(es_model_data_hash(m), es_model_data_hash(m2)) == 0;
  es_model_free(m2);
  es_model_synthetic(2, 0, &m2);
  int differs = m2 && memcmp(m->arena, m2->arena, m->arena_bytes) != 0;
  check_true("synthetic_deterministic_per_seed", same && differs);
  check_true("synthetic_hashes_and_latency", strcmp(es_model_config_hash(m), ES_CONFIG_HASH_HEX) == 0 &&
                                             strlen(es_model_data_hash(m)) == 64 &&
                                             es_model_latency_samples(m) == 128);
  es_model_free(m2);

  /* All-pass: for any decoder input, 1.5 tanh(fc2 output) is 1 on the real tap-0 entries and 0
     elsewhere. Every other tensor equals the allpass = 0 model of the same seed. */
  es_model_synthetic(1, 1, &ap);
  double worst = 0;
  float h[128], y[198];
  fill_rand(h, 128, 1.0f);
  for (int k = 0; k < 16; k++) {
    int W = ap->band_width[k], n_coef = 2 * W * ap->band_taps[k];
    es_linear(h, 1, 128, n_coef, ap->dec[k].fc2.w, ap->dec[k].fc2.b, y);
    for (int j = 0; j < n_coef; j++) {
      double coef = 1.5 * tanhf(y[j]), want = j < W ? 1.0 : 0.0;
      worst = fmax(worst, fabs(coef - want));
    }
  }
  check_le("synthetic_allpass_coefficients_1_on_tap0", worst, 1e-6);
  check_true("synthetic_allpass_other_tensors_unchanged",
             memcmp(ap->enc_lin[0].w, m->enc_lin[0].w, sizeof(float) * 24 * 64) == 0 &&
             memcmp(ap->dec[15].fc1.w, m->dec[15].fc1.w, sizeof(float) * 64 * 128) == 0 &&
             memcmp(ap->tok_g.b, m->tok_g.b, sizeof(float) * 128) == 0);
  es_model_free(ap);
  es_model_free(m);
  es_model_free(NULL);                             /* allowed, does nothing */
}

/* ---------------------------------------------------------------------------------------------- */
/* Package E-1 rows: the stream, attention, deep filter, pooling, enrollment and conditioning     */
/* (20-engine 14.1), all on synthetic models                                                      */
/* ---------------------------------------------------------------------------------------------- */

/* A conditioning set with n tokens of random keys and values and small random FiLM values. */
static es_cond *random_cond(int n) {
  es_cond *c = es_cond_new();
  fill_rand(c->film, 4096, 0.1f);
  fill_rand(&c->k[0][0][0], 4 * 68 * 16, 1.0f);
  fill_rand(&c->v[0][0][0], 4 * 68 * 16, 1.0f);
  c->n_tokens = n;
  return c;
}

/* 1 if every byte of the stream state is zero. */
static int state_is_zero(const es_stream *s) {
  const unsigned char *p = (const unsigned char *)&s->st;
  for (size_t i = 0; i < sizeof(s->st); i++)
    if (p[i]) return 0;
  return 1;
}

static void test_stream(void) {
  enum { HOPS = 200, N = 64 * HOPS };
  static float x[N], y[N], y2[N], hd[2 * HOPS], hd2[2 * HOPS];
  fill_rand(x, N, 0.5f);

  /* Model-level all-pass (14.1 STFT row): the synthetic all-pass model makes every coefficient 1 on
     tap 0, so the stream must give the input back delayed by 64 samples (hop 0 emits silence). */
  es_model *ap = NULL;
  es_model_synthetic(3, 1, &ap);
  es_stream *s = es_stream_new(ap);
  es_cond *c4 = es_cond_new();
  c4->n_tokens = 4;                                /* any valid set: the decoder ignores z here */
  for (int j = 0; j < HOPS; j++) es_stream_hop(s, c4, x + 64 * j, y + 64 * j, hd + 2 * j);
  double err = 0;
  for (int n = 0; n < N; n++) err = fmax(err, fabs(y[n] - (n >= 64 ? x[n - 64] : 0.0f)));
  check_le("stream_synthetic_allpass_model_delay64", err, 1e-5);
  es_stream_free(s);
  es_model_free(ap);

  /* Determinism (14.1): 200 hops twice from reset with a random model and conditioning, memcmp. */
  es_model *m = NULL;
  es_model_synthetic(4, 0, &m);
  es_cond *c = random_cond(37);
  s = es_stream_new(m);
  for (int j = 0; j < HOPS; j++) es_stream_hop(s, c, x + 64 * j, y + 64 * j, hd + 2 * j);
  es_stream_reset(s);
  int resets = 0;
  for (int j = 0; j < HOPS; j++) resets += es_stream_hop(s, c, x + 64 * j, y2 + 64 * j, hd2 + 2 * j);
  int finite = 1;
  for (int n = 0; n < N; n++) finite &= isfinite(y[n]) != 0;
  check_true("stream_determinism_200_hops_memcmp", memcmp(y, y2, sizeof(y)) == 0 && memcmp(hd, hd2, sizeof(hd)) == 0 &&
                                                   resets == 0 && finite);

  /* NaN guard (14.1): a NaN in the analysis history makes the hop return 1 with zero output and heads,
     and leaves the whole state zero. */
  float out[64], h[2];
  s->st.ana[100] = NAN;
  int ret = es_stream_hop(s, c, x, out, h);
  int zero = h[0] == 0 && h[1] == 0;
  for (int i = 0; i < 64; i++) zero &= out[i] == 0.0f;
  check_true("stream_nan_guard_resets", ret == 1 && zero && state_is_zero(s));

  /* An empty or missing conditioning set (n_tokens < 4) outputs silence and leaves the state alone. */
  es_stream_hop(s, c, x, out, h);                  /* make the state non-zero again */
  es_stream_state before = s->st;
  es_cond *empty = es_cond_new();                  /* n_tokens = 0 */
  int r1 = es_stream_hop(s, empty, x, out, h), r2 = es_stream_hop(s, NULL, x, out, h);
  zero = h[0] == 0 && h[1] == 0;
  for (int i = 0; i < 64; i++) zero &= out[i] == 0.0f;
  check_true("stream_empty_cond_silent_state_kept", r1 == 0 && r2 == 0 && zero &&
                                                    memcmp(&before, &s->st, sizeof(before)) == 0);
  es_cond_free(empty);
  es_cond_free(c4);
  es_cond_free(c);
  es_stream_free(s);
  es_model_free(m);
  es_stream_free(NULL);                            /* allowed, does nothing */
  check_true("stream_new_null_model", es_stream_new(NULL) == NULL);
}

/* Attention of 20-engine 6.4 vs naive double softmax attention, n = 4, 37, 68 tokens. */
static void test_attention(void) {
  es_model *m = NULL;
  es_model_synthetic(5, 0, &m);
  int ns[3] = { 4, 37, 68 };
  double worst = 0;
  for (int t = 0; t < 3; t++) {
    es_cond *c = random_cond(ns[t]);
    float z[1024], q[64], sc[4 * 68], a[64];
    double zd[1024], qd[64], ad[64];
    fill_rand(z, 1024, 1.0f);
    for (int i = 0; i < 1024; i++) zd[i] = z[i];
    for (int e = 0; e < 64; e++) {                 /* q = z Wq + bq */
      double acc = m->att_q.b[e];
      for (int i = 0; i < 1024; i++) acc += (double)z[i] * m->att_q.w[i * 64 + e];
      qd[e] = acc;
    }
    for (int h = 0; h < 4; h++) {                  /* softmax over the n tokens, per head */
      double s[68], mx = -INFINITY, sum = 0;
      for (int j = 0; j < ns[t]; j++) {
        double d = 0;
        for (int e = 0; e < 16; e++) d += qd[16 * h + e] * c->k[h][j][e];
        s[j] = 0.25 * d;
        mx = fmax(mx, s[j]);
      }
      for (int j = 0; j < ns[t]; j++) { s[j] = exp(s[j] - mx); sum += s[j]; }
      for (int e = 0; e < 16; e++) {
        double acc = 0;
        for (int j = 0; j < ns[t]; j++) acc += s[j] / sum * c->v[h][j][e];
        ad[16 * h + e] = acc;
      }
    }
    for (int i = 0; i < 1024; i++) {               /* z += a Wo + bo */
      double acc = m->att_o.b[i];
      for (int e = 0; e < 64; e++) acc += ad[e] * m->att_o.w[e * 1024 + i];
      zd[i] += acc;
    }
    es_attention(m, c, z, q, sc, a);
    worst = fmax(worst, max_abs_diff_fd(z, zd, 1024));
    es_cond_free(c);
  }
  check_le("attention_vs_naive_double_n4_37_68", worst, 1e-5);
  es_model_free(m);
}

/* Deep filter of 20-engine 6.5 vs naive complex double over 3 hops, history shift included. */
static void test_deep_filter(void) {
  es_model *m = NULL;
  es_model_synthetic(6, 0, &m);
  static float coef[3][1026], Xr[3][257], Xi[3][257];
  float xhist[2][128][2], Yr[257], Yi[257];
  memset(xhist, 0, sizeof(xhist));
  fill_rand(&coef[0][0], 3 * 1026, 1.5f);
  fill_rand(&Xr[0][0], 3 * 257, 1.0f);
  fill_rand(&Xi[0][0], 3 * 257, 1.0f);
  double worst = 0;
  for (int t = 0; t < 3; t++) {
    es_deep_filter(m, coef[t], Xr[t], Xi[t], xhist, Yr, Yi);
    for (int k = 0; k < 16; k++) {
      int s0 = m->band_start[k], W = m->band_width[k], T = m->band_taps[k];
      const float *cr = coef[t] + m->coef_off[k], *ci = cr + W * T;
      for (int j = 0; j < W; j++) {
        int f = s0 + j;
        double yr = 0, yi = 0;
        for (int tau = 0; tau < T && tau <= t; tau++) {   /* X before the first hop is 0 */
          double ar = cr[tau * W + j], ai = ci[tau * W + j], xr = Xr[t - tau][f], xi = Xi[t - tau][f];
          yr += ar * xr - ai * xi;
          yi += ar * xi + ai * xr;
        }
        worst = fmax(worst, fmax(fabs(Yr[f] - yr), fabs(Yi[f] - yi)));
      }
    }
  }
  check_le("deep_filter_vs_naive_3_hops", worst, 1e-6);
  es_model_free(m);
}

/* Pooling of 20-engine 7.2 steps 4 and 6: one ordered list of kept frames, groups of 3 from its start,
   remainder dropped; selection floor(i * n_avail / limit). */
static void test_pooling(void) {
  int groups[48][3], nt = 0, kept = 0;
  uint8_t a[12] = { 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1 };   /* kept 0 1 3 6 7 8 9 11 */
  int n_avail = es_token_groups(a, 12, 48, groups, &nt, &kept);
  check_true("pooling_gaps_and_remainder", n_avail == 2 && nt == 2 && kept == 8 &&
             groups[0][0] == 0 && groups[0][1] == 1 && groups[0][2] == 3 &&
             groups[1][0] == 6 && groups[1][1] == 7 && groups[1][2] == 8);

  /* 160 frames, every 16th off: 150 kept frames, n_avail = 50. */
  uint8_t b[160];
  int list[160], cnt = 0;
  for (int f = 0; f < 160; f++) { b[f] = f % 16 != 15; if (b[f]) list[cnt++] = f; }
  int limits[2] = { 48, 16 }, ok = 1;
  for (int L = 0; L < 2; L++) {
    n_avail = es_token_groups(b, 160, limits[L], groups, &nt, &kept);
    ok &= n_avail == 50 && nt == limits[L] && kept == 150;
    for (int i = 0; i < nt; i++) {
      int gi = (int)((int64_t)i * 50 / limits[L]);
      for (int r = 0; r < 3; r++) ok &= groups[i][r] == list[3 * gi + r];
    }
  }
  check_true("pooling_selection_50_to_48_and_16", ok);

  /* Any nonzero byte counts as kept, like vad.py token_groups (np.flatnonzero): the same mask written
     with 2 and 255 instead of 1 gives the same groups and kept count. */
  uint8_t c[12];
  for (int f = 0; f < 12; f++) c[f] = a[f] ? (uint8_t)(f % 2 ? 2 : 255) : 0;
  int groups2[48][3], nt2 = 0, kept2 = 0;
  int n_avail2 = es_token_groups(c, 12, 48, groups2, &nt2, &kept2);
  n_avail = es_token_groups(a, 12, 48, groups, &nt, &kept);
  check_true("pooling_nonzero_mask_counts_as_kept", n_avail2 == n_avail && nt2 == nt && kept2 == kept &&
             memcmp(groups2, groups, sizeof(int[3]) * (size_t)nt) == 0);
}

/* Enrollment (20-engine 7.3) and es_cond_build (7.4) rules on a synthetic model. */
static void test_enroll_cond(void) {
  es_model *m = NULL;
  es_model_synthetic(8, 0, &m);
  enum { N = 16000 };                              /* F = 32 frames */
  static float pt[N], po[N], tone[N];
  float e_t[256], e_o[256];
  uint8_t ones[32];
  memset(ones, 1, sizeof(ones));
  fill_rand(pt, N, 0.3f);
  fill_rand(po, N, 0.3f);
  fill_rand(e_t, 256, 0.1f);
  fill_rand(e_o, 256, 0.1f);

  es_voiceprint *vt = es_voiceprint_new(), *vo = es_voiceprint_new(), *vo2 = es_voiceprint_new();
  es_status st = es_enroll(m, pt, N, ones, ES_ROLE_TARGET, e_t, vt);
  float g[128], ss = 0;
  es_voiceprint_get_g(vt, g);
  for (int i = 0; i < 128; i++) ss += g[i] * g[i];
  check_true("enroll_target_32_frames_10_tokens", st == ES_OK && es_voiceprint_num_tokens(vt) == 10 &&
             es_voiceprint_has_wespk(vt) == 1 && fabsf(es_voiceprint_speech_seconds(vt) - 1.024f) < 1e-6f &&
             fabsf(sqrtf(ss) - 1.0f) < 1e-5f);
  st = es_enroll(m, po, N, ones, ES_ROLE_OWN, NULL, vo);       /* own voice without a WeSpeaker vector */
  es_enroll(m, po, N, ones, ES_ROLE_OWN, e_o, vo2);           /* the same with one */
  check_true("enroll_own_without_wespk", st == ES_OK && vo->n_tok == 10 && vo->has_wespk == 0 && vo2->has_wespk == 1);

  /* Argument and length errors leave `out` untouched. */
  es_voiceprint keep = *vt;
  float bad = pt[100];
  pt[100] = NAN;
  int e1 = es_enroll(m, pt, N, ones, ES_ROLE_TARGET, e_t, vt) == ES_ERR_ARG;
  pt[100] = bad;
  e_t[3] = INFINITY;
  int e2 = es_enroll(m, pt, N, ones, ES_ROLE_TARGET, e_t, vt) == ES_ERR_ARG;
  e_t[3] = 0.05f;
  int e3 = es_enroll(m, pt, N, ones, (es_role)5, e_t, vt) == ES_ERR_ARG;
  int e4 = es_enroll(m, pt, 575, ones, ES_ROLE_TARGET, e_t, vt) == ES_ERR_ENROLL_SHORT;
  uint8_t few[32] = { 0 };
  memset(few, 1, 11);                              /* 11 kept frames = 3 tokens */
  int e5 = es_enroll(m, pt, N, few, ES_ROLE_TARGET, e_t, vt) == ES_ERR_ENROLL_SHORT;
  check_true("enroll_errors_out_untouched", e1 && e2 && e3 && e4 && e5 && memcmp(&keep, vt, sizeof(keep)) == 0);
  es_enroll(m, pt, N, ones, ES_ROLE_TARGET, e_t, vt);          /* e_t[3] changed: re-enroll */

  /* mask NULL runs es_vad; es_embed_g is the same token encoder with the es_vad mask and limit 48. */
  memset(tone, 0, sizeof(tone));
  for (int i = 0; i < N; i++) tone[i] = 0.001f * urand();
  put_tone(tone, N, 4, 20);
  uint8_t vmask[32];
  es_vad(tone, N, vmask);
  es_voiceprint *va = es_voiceprint_new(), *vb = es_voiceprint_new();
  int ok_a = es_enroll(m, tone, N, NULL, ES_ROLE_TARGET, NULL, va) == ES_OK;
  int ok_b = es_enroll(m, tone, N, vmask, ES_ROLE_TARGET, NULL, vb) == ES_OK;
  float g_locked[128];
  int ok_g = es_embed_g(m, tone, N, g_locked) == ES_OK;
  check_true("enroll_null_mask_is_es_vad_and_embed_g_matches", ok_a && ok_b && ok_g && va->n_tok == vb->n_tok &&
             memcmp(va->tokens, vb->tokens, sizeof(va->tokens)) == 0 && memcmp(va->g, g_locked, sizeof(g_locked)) == 0);
  es_voiceprint_free(va);
  es_voiceprint_free(vb);

  /* es_cond_build validation (step 1). */
  es_cond *c = es_cond_new(), *c2 = es_cond_new();
  es_voiceprint *fresh = es_voiceprint_new();
  int v1 = es_cond_build(m, ES_MODE_EVERYONE, vt, vo, 1, c) == ES_ERR_ARG;     /* EVERYONE with a target */
  int v2 = es_cond_build(m, ES_MODE_PERSON, NULL, vo, 1, c) == ES_ERR_ARG;     /* PERSON without one */
  int v3 = es_cond_build(m, ES_MODE_PERSON, vo, NULL, 1, c) == ES_ERR_ARG;     /* own voiceprint as target */
  int v4 = es_cond_build(m, ES_MODE_PERSON, vt, vt, 1, c) == ES_ERR_ARG;       /* target voiceprint as own */
  int v5 = es_cond_build(m, ES_MODE_PERSON, fresh, NULL, 1, c) == ES_ERR_ARG;  /* never enrolled */
  int v6 = es_cond_build(m, (es_mode)7, NULL, NULL, 1, c) == ES_ERR_ARG;
  check_true("cond_build_validation", v1 && v2 && v3 && v4 && v5 && v6);
  es_voiceprint_free(fresh);

  /* Step 2: use_wespk = 1 with the own voiceprint lacking its vector gives BOTH nulls, the same FiLM as
     use_wespk = 0. With both vectors present, use_wespk changes the FiLM (so the check means something). */
  es_cond_build(m, ES_MODE_PERSON, vt, vo, 1, c);
  es_cond_build(m, ES_MODE_PERSON, vt, vo, 0, c2);
  int both_null = memcmp(c->film, c2->film, sizeof(c->film)) == 0;
  es_cond_build(m, ES_MODE_PERSON, vt, vo2, 1, c);
  es_cond_build(m, ES_MODE_PERSON, vt, vo2, 0, c2);
  int differs = memcmp(c->film, c2->film, sizeof(c->film)) != 0;
  check_true("cond_one_vector_missing_gives_both_null", both_null && differs);

  /* Steps 3-6 against naive double: FiLM for PERSON with both vectors, the key / value rows of the
     first target token and the last null token, zero rows past n, n and mode. */
  es_cond_build(m, ES_MODE_PERSON, vt, vo2, 1, c);
  double u[770], hdn[256], worst = 0;
  for (int i = 0; i < 256; i++) { u[i] = vt->e[i]; u[384 + i] = vo2->e[i]; }
  for (int i = 0; i < 128; i++) { u[256 + i] = vt->g[i]; u[640 + i] = vo2->g[i]; }
  u[768] = 1; u[769] = 0;
  for (int o = 0; o < 256; o++) {
    double acc = m->film0.b[o];
    for (int i = 0; i < 770; i++) acc += u[i] * m->film0.w[i * 256 + o];
    hdn[o] = acc / (1.0 + exp(-acc));
  }
  for (int o = 0; o < 4096; o++) {
    double acc = m->film2.b[o];
    for (int i = 0; i < 256; i++) acc += hdn[i] * m->film2.w[i * 4096 + o];
    worst = fmax(worst, fabs(c->film[o] - acc));
  }
  int n = c->n_tokens, rows[2] = { 0, n - 1 };
  for (int r = 0; r < 2; r++) {
    double tokd[64];
    for (int e = 0; e < 64; e++)
      tokd[e] = r == 0 ? (double)vt->tokens[0][e] + m->type_emb[e] : (double)m->null_tok[3 * 64 + e] + m->type_emb[128 + e];
    for (int o = 0; o < 128; o++) {
      double acc = m->att_kv.b[o];
      for (int e = 0; e < 64; e++) acc += tokd[e] * m->att_kv.w[e * 128 + o];
      float got = o < 64 ? c->k[o / 16][rows[r]][o % 16] : c->v[(o - 64) / 16][rows[r]][o % 16];
      worst = fmax(worst, fabs(got - acc));
    }
  }
  int zero_tail = 1;
  for (int h = 0; h < 4; h++)
    for (int j = n; j < 68; j++)
      for (int e = 0; e < 16; e++) zero_tail &= c->k[h][j][e] == 0.0f && c->v[h][j][e] == 0.0f;
  check_le("cond_film_and_kv_vs_naive_double", worst, 1e-5);
  check_true("cond_person_n_mode_zero_tail", n == 10 + 10 + 4 && c->mode == ES_MODE_PERSON && zero_tail);
  es_cond_build(m, ES_MODE_EVERYONE, NULL, vo2, 1, c);
  check_true("cond_everyone_n_mode", c->n_tokens == 10 + 4 && c->mode == ES_MODE_EVERYONE);
  es_cond_build(m, ES_MODE_EVERYONE, NULL, NULL, 1, c);
  check_true("cond_everyone_no_own_4_nulls", c->n_tokens == 4);

  es_cond_free(c);
  es_cond_free(c2);
  es_voiceprint_free(vt);
  es_voiceprint_free(vo);
  es_voiceprint_free(vo2);
  es_model_free(m);
}

static void test_misc(void) {
  es_params p;
  es_params_default(&p);
  check_true("params_default_values",
             p.voice_db == 0 && p.room_db == -60 && p.max_gain_db == 18 && p.out_gain_db == 0 &&
             p.slot_gain[0] == 1 && p.slot_gain[1] == 0 && p.slot_gain[2] == 0 && p.xfade_ms == 150 &&
             p.duck_own == 1 && p.gate == 1 && p.agc == 1 && p.bypass == 0);
  int ok = 1;
  for (int s = 0; s >= -12; s--) {
    const char *txt = es_status_str((es_status)s);
    if (!txt || strcmp(txt, "unknown status") == 0) ok = 0;
  }
  check_true("status_str_all_codes", ok && strcmp(es_status_str((es_status)-99), "unknown status") == 0);
  /* The three atomic types the engine's stats use. The 14.1 row on the real es_engine fields is
     atomics_lock_free_stats_fields (E-2 section below). */
  _Atomic uint64_t a64 = 0;
  _Atomic float af = 0;
  _Atomic int32_t a32 = 0;
  check_true("atomics_lock_free_u64_f32_i32",
             atomic_is_lock_free(&a64) && atomic_is_lock_free(&af) && atomic_is_lock_free(&a32));
}

/* ---------------------------------------------------------------------------------------------- */
/* Package E-2 rows: history ring future cursor, engine create rules, start state, input taps,     */
/* params clamping, post formulas, cues, runaway guard, stats atomics. The heavier engine rows     */
/* (alignment, gain cap, hand-off race, impulse delay, slack step) are in tests/test_latency.c.    */
/* ---------------------------------------------------------------------------------------------- */
static void test_hist_future_cursor(void) {
  enum { HS = 65536 };
  float *hb = calloc(HS, sizeof(float)), chunk[1000], dst[64];
  es_hist h;
  hist_init(&h, hb, HS);
  for (int i = 0; i < 1000; i++) chunk[i] = (float)i;
  hist_write(&h, chunk, 1000);
  uint64_t cursor = 1050;                                 /* 50 ahead of the writer */
  int n = hist_read(&h, &cursor, dst, 64);
  check_true("hist_future_cursor_returns_0_keeps_cursor", n == 0 && cursor == 1050);
  hist_write(&h, chunk, 1000);                            /* the writer passes the cursor */
  n = hist_read(&h, &cursor, dst, 64);
  check_true("hist_future_cursor_reads_once_writer_passes", n == 64 && cursor == 1114 && dst[0] == 50.0f);
  free(hb);
}

/* A synthetic engine driven by this thread (render, then run_hops), unmuted or not. */
static es_engine *make_engine(const es_model *m, int rate, int frames, int n_inst, int slack) {
  es_engine_config cfg = { rate, frames, 4096, n_inst, slack };
  es_engine *e = NULL;
  if (es_engine_new(m, &cfg, &e) != ES_OK) return NULL;
  es_engine_set_manual_worker(e, 1);
  return e;
}

static void test_engine_create_and_taps(void) {
  es_model *m = NULL;
  es_model_synthetic(5, 1, &m);
  es_engine *e = NULL;
  es_engine_config c = { 48000, 64, 4096, 1, 43 };
  es_engine_config bad = c;
  int ok = 1;
  bad.io_rate = 32000;   ok = ok && es_engine_new(m, &bad, &e) == ES_ERR_RATE && e == NULL;
  bad = c; bad.io_frames = 8;    ok = ok && es_engine_new(m, &bad, &e) == ES_ERR_ARG;
  bad = c; bad.max_frames = 5000; ok = ok && es_engine_new(m, &bad, &e) == ES_ERR_ARG;
  bad = c; bad.n_instances = 4;  ok = ok && es_engine_new(m, &bad, &e) == ES_ERR_ARG;
  bad = c; bad.slack16 = 30;     ok = ok && es_engine_new(m, &bad, &e) == ES_ERR_ARG;
  ok = ok && es_engine_new(NULL, &c, &e) == ES_ERR_ARG;
  check_true("engine_new_rejects_bad_config", ok);

  /* P_min inside the engine equals the 20-engine 9.4 table. */
  static const int sizes[13] = { 16, 32, 48, 64, 96, 128, 192, 256, 480, 512, 1024, 1104, 4096 };
  static const int p48[13] = { 64, 64, 64, 64, 64, 86, 64, 128, 192, 214, 384, 416, 1408 };
  static const int p44[13] = { 69, 75, 80, 86, 98, 110, 132, 156, 237, 249, 435, 463, 1549 };
  int bad_n = 0;
  for (int i = 0; i < 13; i++) {
    es_engine *a = make_engine(m, 48000, sizes[i], 1, 22), *b = make_engine(m, 44100, sizes[i], 1, 64);
    if (!a || a->p_min != p48[i] || ring_count(&a->out16) != (uint32_t)(p48[i] + 22)) bad_n++;
    if (!b || b->p_min != p44[i] || ring_count(&b->out16) != (uint32_t)(p44[i] + 64)) bad_n++;
    es_engine_free(a);
    es_engine_free(b);
  }
  check("engine_pmin_and_prefill_match_table", bad_n == 0, bad_n, 0);

  /* Start state: exact zeros until the first set_mute(e, 0), then a 400 ms raised-cosine fade-in. */
  e = make_engine(m, 48000, 64, 1, 43);
  es_params p;
  es_params_default(&p);
  p.bypass = 1;
  es_engine_set_params(e, &p);
  float in[64], out[64];
  int zeros = 1, k = 0;
  for (; k < 7500; k++) {                                 /* 10 s of a 1 kHz tone, muted */
    for (int i = 0; i < 64; i++) in[i] = 0.3f * sinf(2.0f * (float)ES_PI * 1000.0f * (float)(64 * k + i) / 48000.0f);
    es_engine_render(e, in, out, 64);
    es_engine_run_hops(e);
    for (int i = 0; i < 64; i++) zeros = zeros && out[i] == 0.0f;
  }
  check_true("start_state_exact_zeros_until_unmute", zeros);
  es_engine_set_mute(e, 0);
  int full_at = -1;
  for (int j = 0; j < 400 && full_at < 0; j++) {
    es_engine_render(e, in, out, 64);
    es_engine_run_hops(e);
    if (e->mute.gain == 1.0f) full_at = 64 * (j + 1);
  }
  check_le("start_state_fade_in_400ms_samples_off", fabs(full_at - 19200.0), 64);

  /* Input taps: a new tap starts at the write position and gets nothing old; a future cursor
     reads nothing; an old cursor starts at the oldest held sample and counts the skip. Another
     10 s of tone first, so more than the 15.36 s readable span has been written. */
  for (int j = 0; j < 7500; j++) {
    es_engine_render(e, in, out, 64);
    es_engine_run_hops(e);
  }
  uint64_t pos = es_engine_input_position(e);
  float *dst = malloc(sizeof(float) * 300000);
  uint64_t cur = pos;
  int n0 = es_engine_read_input(e, &cur, dst, 4096);
  check_true("input_tap_from_position_reads_nothing_old", pos > 300000 && n0 == 0 && cur == pos);
  uint64_t fut = pos + 50;
  int nf = es_engine_read_input(e, &fut, dst, 4096);
  check_true("input_tap_future_cursor_returns_0_keeps_cursor", nf == 0 && fut == pos + 50);
  es_engine_render(e, in, out, 64);
  int n1 = es_engine_read_input(e, &cur, dst, 4096);
  check_true("input_tap_reads_new_samples", n1 == 21 || n1 == 22);
  uint64_t old = 0;
  int n2 = es_engine_read_input(e, &old, dst, 300000);
  uint64_t w = es_engine_input_position(e), keep = ES_HISTORY_SAMPLES - ES_HIST_GUARD;
  check_true("input_tap_old_cursor_starts_at_oldest_counts_skip",
             n2 == (int)keep && old == w && old - 0 - (uint64_t)n2 == w - keep);
  free(dst);

  /* Stats fields are lock-free atomics (8.7). */
  check_true("atomics_lock_free_stats_fields",
             atomic_is_lock_free(&e->s_hops) && atomic_is_lock_free(&e->s_p_self) &&
             atomic_is_lock_free(&e->s_slack16) && atomic_is_lock_free(&e->active[0]) &&
             atomic_is_lock_free(&e->pwords[0]));

  /* set_params: clamps to the contract ranges, non-finite fields take the default. */
  es_params q;
  es_params_default(&q);
  q.voice_db = 40; q.room_db = NAN; q.max_gain_db = -5; q.out_gain_db = 9; q.slot_gain[1] = 7;
  q.xfade_ms = 0; q.gate = 5; q.bypass = 0;
  es_engine_set_params(e, &q);
  es_params r;
  uint32_t words[ES_PARAM_WORDS];
  for (int i = 0; i < ES_PARAM_WORDS; i++) words[i] = atomic_load(&e->pwords[i]);
  memcpy(&r, words, sizeof(r));
  check_true("set_params_clamps_and_defaults",
             r.voice_db == 6 && r.room_db == -60 && r.max_gain_db == 0 && r.out_gain_db == 0 &&
             r.slot_gain[1] == 1 && r.xfade_ms == 4 && r.gate == 1 && (atomic_load(&e->pseq) & 1u) == 0);
  es_engine_free(e);
  es_model_free(m);
}

/* Post formulas (20-engine 10) on es_post_hop directly, one slot, synthetic heads. */
static void post_run(es_post *p, const es_params *prm, float pt, float ps, float level, int hops, float *y) {
  static float ys[ES_MAX_INSTANCES][ES_HOP], xd[ES_HOP];
  float hd[ES_MAX_INSTANCES][2] = { { pt, ps }, { 0, 0 }, { 0, 0 } };
  int act[ES_MAX_INSTANCES] = { 1, 0, 0 };
  float a, b;
  for (int h = 0; h < hops; h++) {
    for (int i = 0; i < ES_HOP; i++) {
      ys[0][i] = level * sinf(2.0f * (float)ES_PI * (float)(64 * h + i) / 16.0f);   /* 1 kHz */
      xd[i] = ys[0][i];
    }
    es_post_hop(p, prm, 1, (const float (*)[ES_HOP])ys, (const float (*)[2])hd, act, xd, y, &a, &b);
  }
}

static void test_post(void) {
  es_post p;
  es_params prm;
  float y[ES_HOP];
  es_params_default(&prm);
  prm.agc = 0;
  prm.duck_own = 0;
  es_post_init(&p, &prm);
  float a200 = 1.0f - expf(-4.0f / 200.0f), a150 = 1.0f - expf(-4.0f / 150.0f);

  /* Gate: p 0.35 gives -6 dB; from open it closes with 200 ms; p 0.6 opens it in one hop. */
  post_run(&p, &prm, 0.35f, 0.0f, 0.1f, 1, y);
  double e1 = fabs(p.g_db[0] - (0.0f + a200 * (-6.0f - 0.0f)));
  post_run(&p, &prm, 0.35f, 0.0f, 0.1f, 2000, y);
  double e2 = fabs(p.g_db[0] + 6.0f);
  post_run(&p, &prm, 0.1f, 0.0f, 0.1f, 2000, y);
  double e3 = fabs(p.g_db[0] + 12.0f);
  post_run(&p, &prm, 0.6f, 0.0f, 0.1f, 1, y);
  check_le("post_gate_close_200ms_floor_minus12_open_1hop", fmax(fmax(e1, e2), fmax(e3, fabs(p.g_db[0]))), 1e-4);

  /* Duck: -15 dB within one hop while p_self > 0.5, release with 150 ms. */
  prm.duck_own = 1;
  post_run(&p, &prm, 0.9f, 0.9f, 0.1f, 1, y);
  double d1 = fabs(p.d_db + 15.0f);
  post_run(&p, &prm, 0.9f, 0.1f, 0.1f, 1, y);
  double d2 = fabs(p.d_db - (-15.0f + a150 * 15.0f));
  check_le("post_duck_1hop_attack_150ms_release", fmax(d1, d2), 1e-4);

  /* Room envelope: 0 at once while own voice, also with duck_own = 0, recovering with 150 ms. */
  prm.duck_own = 0;
  prm.room_db = -8;
  post_run(&p, &prm, 0.9f, 0.1f, 0.1f, 500, y);
  float env_before = p.r_env;
  post_run(&p, &prm, 0.9f, 0.9f, 0.1f, 1, y);
  float env_own = p.r_env, room_own = p.room_lin;
  post_run(&p, &prm, 0.9f, 0.1f, 0.1f, 1, y);
  check_true("post_room_zero_while_own_even_without_duck",
             env_before > 0.99f && env_own == 0.0f && room_own == 0.0f && fabsf(p.r_env - a150) < 1e-6f);
  prm.room_db = -60;

  /* AGC: never more than 0.024 dB per hop, settles toward -26 dBFS + voice_db, frozen without speech. */
  prm.agc = 1;
  es_post_init(&p, &prm);
  double worst = 0;
  float prev = p.agc_db;
  for (int k = 0; k < 40; k++) {                          /* level jumps every 50 hops */
    float lvl = (k & 1) ? 0.5f : 0.001f;
    for (int h = 0; h < 50; h++) {
      post_run(&p, &prm, 0.9f, 0.0f, lvl, 1, y);
      worst = fmax(worst, fabs(p.agc_db - prev));
      prev = p.agc_db;
    }
  }
  check_le("post_agc_slew_per_hop_db", worst, 0.024 + 1e-6);
  es_post_init(&p, &prm);
  post_run(&p, &prm, 0.9f, 0.0f, 0.01f, 5000, y);          /* -43 dBFS RMS sum: wants +17 dB */
  double settle = fabs(p.agc_db - (-26.0f + 43.0103f));
  float frozen = p.agc_db;
  post_run(&p, &prm, 0.3f, 0.0f, 0.5f, 500, y);            /* loud but no speech: frozen */
  check_le("post_agc_settles_to_target_db", settle, 0.05);
  check_true("post_agc_frozen_without_speech", p.agc_db == frozen);

  /* Bypass: after 5 hops of ramp the output is exactly x_d. */
  prm.bypass = 1;
  post_run(&p, &prm, 0.9f, 0.0f, 0.2f, 6, y);
  int exact = 1;
  for (int i = 0; i < ES_HOP; i++) exact = exact && y[i] == 0.2f * sinf(2.0f * (float)ES_PI * (float)(64 * 5 + i) / 16.0f);
  check_true("post_bypass_output_exactly_xd", exact && p.bm == 1.0f);

  /* Gain cap on steady noise (E-2 review). x_d and the slot output are independent white noise, the
     slot 30x (+29.5 dB) louder; gate, duck, AGC and room off, so the cap works on every hop. Step 5 must
     follow the 20-engine 10 formula exactly (a literal double-precision copy runs beside it on the same
     signals) and must not pump: the gain it applies may move at most 0.5 dB from one hop to the next.
     Measured - the formula alone moves about 0.16 dB; the per-hop bound E-2 first added moved 1.6 dB. */
  es_params q;
  es_params_default(&q);
  q.gate = 0;
  q.duck_own = 0;
  q.agc = 0;
  q.room_db = -60;
  es_post_init(&p, &q);
  {
    static float ys[ES_MAX_INSTANCES][ES_HOP], xd[ES_HOP];
    float hd[ES_MAX_INSTANCES][2] = { { 0.9f, 0.0f }, { 0, 0 }, { 0, 0 } };
    int act[ES_MAX_INSTANCES] = { 1, 0, 0 };
    double a50 = 1.0 - exp(-4.0 / 50.0), a200d = 1.0 - exp(-4.0 / 200.0);
    double in_ms = 0, out_ms = 0, cap_db = 0, ferr = 0, step = 0, g_last = 0;
    uint32_t r = 99u;                                    /* own generator: g_rng stays untouched */
    for (int h = 0; h < 5000; h++) {
      for (int i = 0; i < ES_HOP; i++) {
        r ^= r << 13; r ^= r >> 17; r ^= r << 5;
        xd[i] = 0.01f * ((float)((r >> 8) * (1.0 / 16777216.0)) * 2.0f - 1.0f);
        r ^= r << 13; r ^= r >> 17; r ^= r << 5;
        ys[0][i] = 0.3f * ((float)((r >> 8) * (1.0 / 16777216.0)) * 2.0f - 1.0f);
      }
      float a, b;
      es_post_hop(&p, &q, 1, (const float (*)[ES_HOP])ys, (const float (*)[2])hd, act, xd, y, &a, &b);
      double mx = 0, my = 0;                             /* literal step 5; p.ypre = output of (4) */
      for (int i = 0; i < ES_HOP; i++) {
        mx += (double)xd[i] * xd[i] / ES_HOP;
        my += (double)p.ypre[i] * p.ypre[i] / ES_HOP;
      }
      in_ms += a50 * (mx - in_ms);
      out_ms += a50 * (my - out_ms);
      double ct = -fmax(10.0 * log10(out_ms + 1e-10) - 10.0 * log10(in_ms + 1e-10) - 24.0, 0.0);
      cap_db = ct < cap_db ? ct : cap_db + a200d * (ct - cap_db);
      ferr = fmax(ferr, fabs(cap_db - p.cap_db));
      double eo = 0, ei = 0;                             /* gain applied in the second half of the hop */
      for (int i = 32; i < ES_HOP; i++) {
        eo += (double)y[i] * y[i];
        ei += (double)ys[0][i] * ys[0][i];
      }
      double g = 10.0 * log10(eo / ei);
      if (h > 500) step = fmax(step, fabs(g - g_last));
      g_last = g;
    }
    check_le("post_gain_cap_follows_step5_formula_db", ferr, 1e-3);
    check_le("post_gain_cap_steady_noise_hop_step_db", step, 0.5);
  }
}

/* Cue beeps and the runaway guard (20-engine 11). */
static void test_cue_guard(void) {
  es_cue c;
  es_cue_init(&c, 48000.0f);
  enum { N = 24000 };
  static float y[N];
  memset(y, 0, sizeof(y));
  es_cue_start(&c, 3);
  int total = c.total;
  for (int off = 0; off < N; off += 100) es_cue_add(&c, y + off, 100);
  double peak = 0, gap = 0, after = 0;
  for (int i = 0; i < N; i++) {
    peak = fmax(peak, fabs(y[i]));
    int t = i % 7200;
    if (i < total && t >= 2880) gap = fmax(gap, fabs(y[i]));  /* 60 to 150 ms: silence */
    if (i >= total) after = fmax(after, fabs(y[i]));
  }
  check_true("cue_3_beeps_length_360ms", total == 17280 && c.total == 0);
  check_le("cue_peak_minus18dBFS", fabs(peak - 0.1258925), 2e-4);
  check_true("cue_gaps_and_tail_silent", gap == 0.0 && after == 0.0 && fabs(y[0]) < 1e-6);
  es_cue_start(&c, 9);
  check_true("cue_beeps_clamped_to_3", c.total == 17280);

  es_guard g;
  es_guard_init(&g, 48000.0f);
  float xin[480], out[480];
  for (int i = 0; i < 480; i++) xin[i] = 1e-4f;              /* -80 dBFS mic */
  int k_on = -1, k_off = -1;
  for (int k = 0; k < 100 && k_on < 0; k++) {                /* loud output: +60 dB over the mic */
    for (int i = 0; i < 480; i++) out[i] = 0.1f;
    if (es_guard_process(&g, xin, out, 480)) k_on = k;
  }
  for (int k = 0; k < 20; k++) {                             /* the duck reaches -20 dB within 5 ms */
    for (int i = 0; i < 480; i++) out[i] = 0.1f;
    es_guard_process(&g, xin, out, 480);
  }
  float ducked = out[479];
  int k_quiet = -1;                                          /* first chunk with the condition false */
  for (int k = 0; k < 600 && k_off < 0; k++) {               /* quiet: clears after 2 s of that */
    for (int i = 0; i < 480; i++) out[i] = 0.0f;
    int howl = es_guard_process(&g, xin, out, 480);
    if (k_quiet < 0 && g.over == 0) k_quiet = k;
    if (!howl) k_off = k;
  }
  /* howl after 0.2 s of loud output: 20 chunks of 10 ms (the 50 ms mean square passes 1e-6 at once). */
  check_true("guard_howl_after_200ms", k_on == 19);
  check_le("guard_duck_minus20dB", fabs(ducked - 0.01f), 1e-6);
  check_true("guard_clears_after_2s_quiet", k_quiet > 0 && k_off == k_quiet + 199);
}

/* ---------------------------------------------------------------------------------------------- */
/* Latency probe (20-engine 12 and 14.1 "Probe"): GCC-PHAT through a simulated echo loop           */
/* ---------------------------------------------------------------------------------------------- */
/* One simulated loop. The engine's input at I/O sample n is
     0.1 * out[n - D]   (its own output D samples earlier, -20 dB)   + white noise at -50 dBFS RMS,
   which is what a speaker-to-mic path with delay D looks like. Extras for the n_ok and spread rows:
   cut_s > 0 cuts the loop (gain 0) for the first cut_s seconds after the probe starts, and D2 > 0
   switches the delay to D2 from d2_s seconds after the start. D = 0 means noise only. */
typedef struct {
  int rate, frames;   /* I/O rate and callback size (frames <= D, so out[n - D] already exists) */
  int D, D2;
  double d2_s, cut_s;
  double noise_dbfs;  /* RMS of the white noise; 0 means the default -50 dBFS */
} loop_case;

/* Median psr of the 5 sweeps of the last capture with GCC-PHAT weighting only lo_hz..hi_hz, and in
   *n_ok how many of them reach the 10 dB validity threshold. */
static float probe_band_psr(const es_engine *e, double lo_hz, double hi_hz, int *n_ok) {
  float lag[5], psr[5];
  *n_ok = 0;
  if (es_probe_analyze_band(e, lo_hz, hi_hz, lag, psr) != ES_OK) return -999.0f;
  for (int i = 0; i < 5; i++) *n_ok += psr[i] >= 10.0f;
  for (int i = 1; i < 5; i++)                       /* insertion sort, then the middle one */
    for (int j = i; j > 0 && psr[j - 1] > psr[j]; j--) { float t = psr[j]; psr[j] = psr[j - 1]; psr[j - 1] = t; }
  return psr[2];
}

/* Runs the loop, starts the probe after 0.5 s (past the 400 ms fade-in), keeps calling
   es_engine_probe_result until it stops saying ES_ERR_STATE. Returns that status and fills *r.
   *api_ok checks the call rules on the way: STATE before any start, BUSY for a second start while
   running, STATE while running, the same cached answer on a second result call, probe_runs == 1, and
   a new start clears the cached answer (STATE again until that run finishes). */
/* band_cmp, when not NULL, receives the median psr of the same capture analyzed over the full band
   ([0], the E-2 weighting) and over 300 Hz to 6 kHz ([1], the engine's weighting since E-3), and the
   number of valid sweeps (psr >= 10 dB) of each ([2] and [3]). */
static es_status probe_loop(const es_model *m, const loop_case *lc, es_probe_result *r, int *api_ok,
                            float *band_cmp) {
  es_engine *e = make_engine(m, lc->rate, lc->frames, 1, 43);
  if (!e) return ES_ERR_NOMEM;
  const int F = lc->frames, pre = lc->rate / 2;
  const int total = pre + (int)(4.6 * lc->rate) + F;
  float *outs = calloc((size_t)total + (size_t)F, sizeof(float));   /* every output sample, for the loop */
  float *in = malloc(sizeof(float) * (size_t)F);
  const double noise_db = lc->noise_dbfs != 0.0 ? lc->noise_dbfs : -50.0;
  const float noise_amp = (float)pow(10.0, noise_db / 20.0) * 1.7320508f;   /* uniform [-a, a]: RMS a / sqrt(3) */
  es_probe_result tmp;
  int ok = es_engine_probe_result(e, &tmp) == ES_ERR_STATE;          /* never started */
  es_engine_set_mute(e, 0);
  es_status st = ES_ERR_STATE;
  int started = -1;
  for (int t = 0; t < total; t += F) {
    if (t >= pre && started < 0) {
      ok = ok && es_engine_probe_start(e) == ES_OK;
      ok = ok && es_engine_probe_start(e) == ES_ERR_BUSY;              /* already running */
      ok = ok && es_engine_probe_result(e, &tmp) == ES_ERR_STATE;      /* not finished */
      started = t;                                                    /* capture index 0 = sample t */
    }
    for (int i = 0; i < F; i++) {
      int n = t + i;
      double rel = started < 0 ? -1.0 : (double)(n - started) / lc->rate;   /* seconds since start */
      int d = (lc->D2 > 0 && rel >= lc->d2_s) ? lc->D2 : lc->D;
      float loop = 0.0f;
      if (d > 0 && n - d >= 0 && !(rel >= 0.0 && rel < lc->cut_s)) loop = 0.1f * outs[n - d];
      in[i] = loop + noise_amp * urand();
    }
    es_engine_render(e, in, outs + t, F);
    es_engine_run_hops(e);
    if (started >= 0 && (st = es_engine_probe_result(e, r)) != ES_ERR_STATE) break;
  }
  if (band_cmp) {                                                     /* before the restart below */
    int n_full = 0, n_band = 0;
    band_cmp[0] = probe_band_psr(e, 0.0, 0.5 * lc->rate, &n_full);
    band_cmp[1] = probe_band_psr(e, 300.0, 6000.0, &n_band);
    band_cmp[2] = (float)n_full;
    band_cmp[3] = (float)n_band;
  }
  es_probe_result again;
  es_status st2 = es_engine_probe_result(e, &again);                   /* cached */
  ok = ok && st2 == st && memcmp(&again, r, sizeof(again)) == 0;
  es_stats s;
  es_engine_get_stats(e, &s);
  ok = ok && s.probe_runs == 1;
  ok = ok && es_engine_probe_start(e) == ES_OK && es_engine_probe_result(e, &tmp) == ES_ERR_STATE;
  *api_ok = ok;
  free(outs);
  free(in);
  es_engine_free(e);
  return st;
}

static void test_probe(void) {
  es_model *m = NULL;
  es_model_synthetic(12, 0, &m);
  char name[96];
  int api_all = 1;

  /* Known loop delays: 20-engine 12's D = 480, 2,400 and 9,600 at 48 kHz, plus 37.5 ms and 120 ms, plus
     one 44.1 kHz route with 441-frame callbacks. Each needs |loop_ms - D / rate| <= 0.05 ms, n_ok = 5. */
  static const loop_case delays[6] = {
    { 48000, 64, 480, 0, 0, 0, 0 }, { 48000, 64, 2400, 0, 0, 0, 0 }, { 48000, 64, 9600, 0, 0, 0, 0 },
    { 48000, 64, 1800, 0, 0, 0, 0 }, { 48000, 64, 5760, 0, 0, 0, 0 }, { 44100, 441, 2205, 0, 0, 0, 0 },
  };
  for (int i = 0; i < 6; i++) {
    es_probe_result r;
    int api = 0;
    es_status st = probe_loop(m, &delays[i], &r, &api, NULL);
    api_all = api_all && api;
    double want = 1000.0 * delays[i].D / delays[i].rate;
    printf("INFO probe_loop rate %d D %d: status %d loop_ms %.4f (want %.4f) spread_ms %.4f psr_db %.2f n_ok %d\n",
           delays[i].rate, delays[i].D, (int)st, r.loop_ms, want, r.spread_ms, r.psr_db, r.n_ok);
    snprintf(name, sizeof(name), "probe_loop_%dk_D%d_err_ms", delays[i].rate / 1000, delays[i].D);
    check_le(name, st == ES_OK ? fabs(r.loop_ms - want) : 1e9, 0.05);
    snprintf(name, sizeof(name), "probe_loop_%dk_D%d_n_ok_5", delays[i].rate / 1000, delays[i].D);
    check(name, st == ES_OK && r.n_ok == 5, r.n_ok, 5);
  }

  /* Noise only (no loop): PROBE_WEAK, and the fields are still filled with finite numbers. */
  loop_case noise = { 48000, 64, 0, 0, 0, 0, 0 };
  es_probe_result rn;
  int api = 0;
  es_status sn = probe_loop(m, &noise, &rn, &api, NULL);
  api_all = api_all && api;
  printf("INFO probe_noise_only: status %d loop_ms %.3f spread_ms %.3f psr_db %.2f n_ok %d\n",
         (int)sn, rn.loop_ms, rn.spread_ms, rn.psr_db, rn.n_ok);
  check_true("probe_noise_only_is_probe_weak_with_spread_filled",
             sn == ES_ERR_PROBE_WEAK && rn.n_ok < 3 && isfinite(rn.spread_ms) && rn.spread_ms >= 0.0f &&
             isfinite(rn.loop_ms) && isfinite(rn.psr_db));

  /* PROBE_WEAK depends only on n_ok: the loop cut for the first 1.6 s hides sweeps 0 and 1 (n_ok 3,
     ES_OK); cut for 2.4 s it also hides sweep 2 (n_ok 2, PROBE_WEAK, spread still filled). */
  loop_case cut3 = { 48000, 64, 2400, 0, 0, 1.6, 0 }, cut2 = { 48000, 64, 2400, 0, 0, 2.4, 0 };
  es_probe_result r3, r2;
  es_status s3 = probe_loop(m, &cut3, &r3, &api, NULL);
  api_all = api_all && api;
  es_status s2 = probe_loop(m, &cut2, &r2, &api, NULL);
  api_all = api_all && api;
  printf("INFO probe_cut_1.6s: status %d n_ok %d loop_ms %.4f; cut_2.4s: status %d n_ok %d loop_ms %.4f spread %.4f\n",
         (int)s3, r3.n_ok, r3.loop_ms, (int)s2, r2.n_ok, r2.loop_ms, r2.spread_ms);
  check_true("probe_n_ok_3_is_ok", s3 == ES_OK && r3.n_ok == 3 && fabs(r3.loop_ms - 50.0) <= 0.05);
  check_true("probe_n_ok_2_is_probe_weak_spread_filled",
             s2 == ES_ERR_PROBE_WEAK && r2.n_ok == 2 && fabs(r2.loop_ms - 50.0) <= 0.05 &&
             r2.spread_ms >= 0.0f && r2.spread_ms <= 0.1f);

  /* The spread is max - min of the valid lags and never makes the engine fail (the app judges it by
     route): 50 ms for sweeps 0..2, 60 ms for sweeps 3 and 4 -> loop 50 ms, spread 10 ms, ES_OK. */
  loop_case two = { 48000, 64, 2400, 2880, 2.4, 0, 0 };
  es_probe_result rt;
  es_status stt = probe_loop(m, &two, &rt, &api, NULL);
  api_all = api_all && api;
  printf("INFO probe_two_delays: status %d n_ok %d loop_ms %.4f spread_ms %.4f\n", (int)stt, rt.n_ok, rt.loop_ms,
         rt.spread_ms);
  check_true("probe_spread_is_max_minus_min_and_not_an_error",
             stt == ES_OK && rt.n_ok == 5 && fabs(rt.loop_ms - 50.0) <= 0.05 && fabs(rt.spread_ms - 10.0) <= 0.1);

  /* GCC-PHAT band limit (DECISIONS L 2026-11-08). Same loop with white noise at -36 dBFS RMS, so the echo
     of the -30 dBFS sweep (taken at -20 dB) sits about 17 dB under the noise. The same capture is analyzed
     twice: full band (the E-2 weighting: every bin at weight 1) and 300 Hz to 6 kHz (the engine now).
     The band limit must raise the median psr by at least 3 dB and find more valid sweeps; the engine's
     own result must stay ES_OK (n_ok >= 3) within 0.05 ms. At the default -50 dBFS the rows above show
     the same gain (psr about 25 dB now against about 17.5 dB with the full band). */
  loop_case loud = { 48000, 64, 2400, 0, 0, 0, -36.0 };
  es_probe_result rl;
  float cmp[4] = { 0, 0, 0, 0 };
  es_status sl = probe_loop(m, &loud, &rl, &api, cmp);
  api_all = api_all && api;
  printf("INFO probe_noise_-36dBFS: full band psr %.2f dB (%d valid sweeps), 300 Hz-6 kHz psr %.2f dB (%d valid); "
         "engine status %d n_ok %d loop_ms %.4f (want 50) psr_db %.2f\n", cmp[0], (int)cmp[2], cmp[1], (int)cmp[3],
         (int)sl, rl.n_ok, rl.loop_ms, rl.psr_db);
  check("probe_band_limited_psr_gain_over_full_band_at_-36dBFS_db", cmp[1] - cmp[0] >= 3.0, cmp[1] - cmp[0], 3.0);
  check("probe_band_limited_more_valid_sweeps_than_full_band_at_-36dBFS", cmp[3] > cmp[2], cmp[3], cmp[2] + 1);
  check_le("probe_loop_48k_D2400_-36dBFS_err_ms", sl == ES_OK ? fabs(rl.loop_ms - 50.0) : 1e9, 0.05);

  /* E-3 fix pass (review): a second noise level, -40 dBFS, so the band-limit rows do not rest on one
     value near the 10 dB validity threshold. The -36 dBFS numbers move a little with the hop path (the
     noise also passes through the model output in this loop): 11.00 dB / 4 valid on the E-3 xt build,
     10.51 dB / 5 valid on the sgemv default. Same checks: gain >= 3 dB and the engine ES_OK within 0.05 ms. */
  loop_case mid = { 48000, 64, 2400, 0, 0, 0, -40.0 };
  es_probe_result rm;
  float cmp40[4] = { 0, 0, 0, 0 };
  es_status sm = probe_loop(m, &mid, &rm, &api, cmp40);
  api_all = api_all && api;
  printf("INFO probe_noise_-40dBFS: full band psr %.2f dB (%d valid sweeps), 300 Hz-6 kHz psr %.2f dB (%d valid); "
         "engine status %d n_ok %d loop_ms %.4f (want 50) psr_db %.2f\n", cmp40[0], (int)cmp40[2], cmp40[1],
         (int)cmp40[3], (int)sm, rm.n_ok, rm.loop_ms, rm.psr_db);
  check("probe_band_limited_psr_gain_over_full_band_at_-40dBFS_db", cmp40[1] - cmp40[0] >= 3.0, cmp40[1] - cmp40[0], 3.0);
  check_le("probe_loop_48k_D2400_-40dBFS_err_ms", sm == ES_OK ? fabs(rm.loop_ms - 50.0) : 1e9, 0.05);

  check_true("probe_call_rules_state_busy_cache_restart", api_all);
  es_model_free(m);
}

/* ---------------------------------------------------------------------------------------------- */
/* WAV reader and writer (tools/wav.c, 20-engine 1 and 14.7)                                       */
/* ---------------------------------------------------------------------------------------------- */
/* Builds a WAV in buf by hand: fmt (16 bytes, or 40 for extensible with the real tag in the GUID),
   an optional LIST chunk of odd size (so the pad byte is exercised), then data. Returns the size. */
static size_t build_wav(uint8_t *buf, int tag, int ext, int ch, int rate, int bits, int list,
                        const uint8_t *data, uint32_t data_len) {
  size_t p = 12;
  uint32_t fmt_len = ext ? 40 : 16, align = (uint32_t)(ch * bits / 8);
  memcpy(buf, "RIFF", 4);
  memcpy(buf + 8, "WAVE", 4);
  memcpy(buf + p, "fmt ", 4);
  put_u32(buf + p + 4, fmt_len);
  uint8_t *f = buf + p + 8;
  memset(f, 0, fmt_len);
  f[0] = (uint8_t)(ext ? 0xFE : tag); f[1] = (uint8_t)(ext ? 0xFF : 0);
  f[2] = (uint8_t)ch;
  put_u32(f + 4, (uint32_t)rate);
  put_u32(f + 8, (uint32_t)rate * align);
  f[12] = (uint8_t)align;
  f[14] = (uint8_t)bits;
  if (ext) { f[16] = 22; f[24] = (uint8_t)tag; }    /* cbSize 22, sub-format tag (GUID bytes 0..1) */
  p += 8 + fmt_len;
  if (list) {                                        /* 3-byte LIST chunk + 1 pad byte */
    memcpy(buf + p, "LIST", 4);
    put_u32(buf + p + 4, 3);
    memcpy(buf + p + 8, "abc\0", 4);
    p += 12;
  }
  memcpy(buf + p, "data", 4);
  put_u32(buf + p + 4, data_len);
  memcpy(buf + p + 8, data, data_len);
  p += 8 + data_len;
  put_u32(buf + 4, (uint32_t)(p - 8));
  return p;
}

static void test_wav(void) {
  enum { N = 1000 };
  static float x[N];
  for (int i = 0; i < N; i++) x[i] = 0.9f * sinf(0.05f * (float)i) + 1e-4f * urand();
  x[0] = 1.0f; x[1] = -1.0f; x[2] = 0.0f;

  /* float32 round trip is exact; PCM16 is within 1.5 / 32768 (32767 on write, 32768 on read). */
  size_t len = 0;
  float *y = NULL;
  int n = 0, rate = 0;
  uint8_t *b = wav_encode(x, N, 16000, 0, &len);
  int ok = b && wav_decode(b, len, &y, &n, &rate, NULL, 0) == 0 && n == N && rate == 16000 &&
           memcmp(x, y, sizeof(x)) == 0 && len == 44 + 4 * N;
  check_true("wav_float32_roundtrip_exact", ok);
  free(b); free(y); y = NULL;
  b = wav_encode(x, N, 48000, 1, &len);
  double err16 = 1e9;
  if (b && wav_decode(b, len, &y, &n, &rate, NULL, 0) == 0 && n == N && rate == 48000) {
    err16 = 0;
    for (int i = 0; i < N; i++) err16 = fmax(err16, fabs((double)y[i] - x[i]));
  }
  check_le("wav_pcm16_roundtrip_err", err16, 1.5 / 32768.0);
  free(b); free(y); y = NULL;
  float clip[3] = { 2.0f, -3.0f, NAN };
  b = wav_encode(clip, 3, 8000, 1, &len);
  ok = b && wav_decode(b, len, &y, &n, &rate, NULL, 0) == 0 && n == 3 &&
       y[0] == 32767.0f / 32768.0f && y[1] == -32767.0f / 32768.0f && y[2] == 0.0f;
  check_true("wav_pcm16_clips_and_nan_to_zero", ok);
  free(b); free(y); y = NULL;

  /* PCM24 stereo: channel 0 only, sign extension, a LIST chunk with a pad byte before data. */
  static uint8_t buf[4096];
  uint8_t d24[18] = { 0xFF, 0xFF, 0x7F, 1, 2, 3,     /* ch0 = +8388607, ch1 = junk */
                      0x00, 0x00, 0x80, 4, 5, 6,     /* ch0 = -8388608 */
                      0x01, 0x00, 0x00, 7, 8, 9 };   /* ch0 = 1 */
  size_t L = build_wav(buf, 1, 0, 2, 44100, 24, 1, d24, sizeof(d24));
  ok = wav_decode(buf, L, &y, &n, &rate, NULL, 0) == 0 && n == 3 && rate == 44100 &&
       y[0] == 8388607.0f / 8388608.0f && y[1] == -1.0f && y[2] == 1.0f / 8388608.0f;
  check_true("wav_pcm24_stereo_channel0_list_chunk", ok);
  free(y); y = NULL;

  /* WAVE_FORMAT_EXTENSIBLE float32 stereo. */
  float fs[4] = { 0.25f, 9.0f, -0.5f, 9.0f };
  L = build_wav(buf, 3, 1, 2, 48000, 32, 0, (const uint8_t *)fs, sizeof(fs));
  ok = wav_decode(buf, L, &y, &n, &rate, NULL, 0) == 0 && n == 2 && y[0] == 0.25f && y[1] == -0.5f;
  check_true("wav_extensible_float32_channel0", ok);
  free(y); y = NULL;

  /* Malformed files fail with a reason and leave *pcm NULL. */
  char err[96];
  int bad = 0;
  uint8_t d16[4] = { 0, 0, 0, 0 };
  L = build_wav(buf, 1, 0, 1, 16000, 16, 0, d16, 4);
  bad += wav_decode(buf, 11, &y, &n, &rate, err, sizeof(err)) == -1 && !y;            /* truncated */
  buf[0] = 'X';
  bad += wav_decode(buf, L, &y, &n, &rate, err, sizeof(err)) == -1 && !y;             /* bad magic */
  buf[0] = 'R';
  bad += wav_decode(buf, L - 12, &y, &n, &rate, err, sizeof(err)) == -1 && !y;        /* no data chunk */
  L = build_wav(buf, 1, 0, 1, 16000, 8, 0, d16, 4);
  bad += wav_decode(buf, L, &y, &n, &rate, err, sizeof(err)) == -1 && !y;             /* 8-bit PCM */
  L = build_wav(buf, 1, 0, 1, 16000, 16, 0, d16, 4);
  buf[12 + 8 + 12] = 3;                                                               /* align 3 */
  bad += wav_decode(buf, L, &y, &n, &rate, err, sizeof(err)) == -1 && !y;
  check("wav_rejects_5_malformed_files", bad == 5, bad, 5);

  /* wav_write + wav_read through a real file in TMPDIR (removed afterwards). */
  const char *tmpdir = getenv("TMPDIR");
  char path[512];
  snprintf(path, sizeof(path), "%s/es_test_wav_XXXXXX", tmpdir && *tmpdir ? tmpdir : "/tmp");
  int fd = mkstemp(path);
  ok = 0;
  if (fd >= 0) {
    close(fd);
    ok = wav_write(path, x, N, 44100, 0, err, sizeof(err)) == 0 &&
         wav_read(path, &y, &n, &rate, err, sizeof(err)) == 0 && n == N && rate == 44100 &&
         memcmp(x, y, sizeof(x)) == 0;
    unlink(path);
  }
  check_true("wav_write_read_file_roundtrip", ok);
  free(y);
}

int main(void) {
  printf("test_units (packages E-0, E-1, E-2)\n");
  test_config();
  test_windows();
  test_fft();
  test_stft();
  test_features();
  test_kernels();
  test_vad();
  test_json();
  test_container();
  test_rings();
  test_resampler();
  test_ramps();
  test_limiter_softclip();
  test_model_synthetic();
  test_stream();
  test_attention();
  test_deep_filter();
  test_pooling();
  test_enroll_cond();
  test_misc();
  test_hist_future_cursor();
  test_engine_create_and_taps();
  test_post();
  test_cue_guard();
  test_probe();
  test_wav();
  printf("test_units: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
