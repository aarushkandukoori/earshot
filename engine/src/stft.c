/* stft.c - windows, the vDSP FFT wrapper, input features and streaming synthesis (20-engine 5).
 *
 * Implements contract 2.1 (signal and framing) and 2.2 row 1 (features).
 * Threads: es_make_windows, es_fft_init/free and the offline helpers are non-RT (they may allocate).
 * es_rfft, es_irfft, es_features, es_analysis_hop and es_synthesis_hop are RT-safe (no allocation,
 * no locks) as long as each thread uses its own es_fft.
 *
 * FFT scaling (the reason for the 0.5 and 1/512 below). vDSP's real FFT returns twice the
 * mathematical DFT in the forward direction, and its inverse returns the plain (unnormalized) inverse
 * sum. torch.fft.rfft is the plain DFT and torch.fft.irfft divides by N = 512, so we multiply the
 * forward result by 0.5 and the inverse result by 1/512. test_units checks both against a float64 DFT.
 *
 * vDSP packs a real FFT of 512 points into 256 complex numbers: re[0] holds bin 0 (DC) and im[0]
 * holds bin 256 (Nyquist), both of which are purely real; re[k], im[k] hold bin k for 1 <= k <= 255.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

/* Periodic Hann window value H_L[n] = 0.5 - 0.5 cos(2 pi n / L), in double. */
static double hann(int n, int L) {
  return 0.5 - 0.5 * cos(2.0 * ES_PI * (double)n / (double)L);
}

/* Analysis and synthesis windows of contract 2.1 / 20-engine 5.1.
   wa = sqrt(H_896) for n < 448 (a slow rise), then sqrt(H_128[n - 384]) (a fast fall).
   ws = 0 for n < 384, then H_128[n - 384] / wa[n], so wa * ws = H_128 on the last 128 samples. */
void es_make_windows(float wa[ES_NFFT], float ws[ES_NFFT]) {
  for (int n = 0; n < ES_NFFT; n++) {
    double a;
    if (n < ES_ANA_RISE) a = sqrt(hann(n, 2 * ES_ANA_RISE));
    else                 a = sqrt(hann(n - ES_SYNTH_START, ES_SYNTH_LEN));
    wa[n] = (float)a;
    if (n < ES_SYNTH_START) ws[n] = 0.0f;
    else                    ws[n] = (float)(hann(n - ES_SYNTH_START, ES_SYNTH_LEN) / a);
  }
}

es_status es_fft_init(es_fft *f) {
  memset(f, 0, sizeof(*f));
  f->setup = vDSP_create_fftsetup(9, kFFTRadix2);   /* 2^9 = 512 points */
  return f->setup ? ES_OK : ES_ERR_NOMEM;
}

void es_fft_free(es_fft *f) {
  if (f->setup) vDSP_destroy_fftsetup(f->setup);
  f->setup = NULL;
}

/* Forward real FFT: x[512] -> X[257], equal to torch.fft.rfft (unnormalized). */
void es_rfft(es_fft *f, const float x[ES_NFFT], float Xr[ES_NBINS], float Xi[ES_NBINS]) {
  DSPSplitComplex sp = { f->re, f->im };
  vDSP_ctoz((const DSPComplex *)x, 2, &sp, 1, 256);      /* even samples to re, odd samples to im */
  vDSP_fft_zrip(f->setup, &sp, 1, 9, kFFTDirection_Forward);
  Xr[0] = 0.5f * f->re[0];   Xi[0] = 0.0f;                 /* DC */
  Xr[256] = 0.5f * f->im[0]; Xi[256] = 0.0f;               /* Nyquist */
  for (int k = 1; k < 256; k++) {
    Xr[k] = 0.5f * f->re[k];
    Xi[k] = 0.5f * f->im[k];
  }
}

/* Inverse real FFT: Y[257] -> y[512], equal to torch.fft.irfft(n=512).
   The imaginary parts of bins 0 and 256 are ignored, as irfft does. */
void es_irfft(es_fft *f, const float Yr[ES_NBINS], const float Yi[ES_NBINS], float y[ES_NFFT]) {
  DSPSplitComplex sp = { f->re, f->im };
  f->re[0] = Yr[0];
  f->im[0] = Yr[256];
  for (int k = 1; k < 256; k++) {
    f->re[k] = Yr[k];
    f->im[k] = Yi[k];
  }
  vDSP_fft_zrip(f->setup, &sp, 1, 9, kFFTDirection_Inverse);
  vDSP_ztoc(&sp, 1, (DSPComplex *)y, 2, 256);              /* re to even samples, im to odd samples */
  float scale = 1.0f / 512.0f;
  vDSP_vsmul(y, 1, &scale, y, 1, ES_NFFT);
}

/* Features of one frame (contract 2.2 row 1, 20-engine 5.3), written bin-major: feat[3*bin + f] with
   f = 0: Re c, f = 1: Im c, f = 2: 0.1 ln(|X|^2 + 1e-8), where c = X * (|X| + 1e-8)^(-0.7).
   work holds three 257-float scratch arrays: p = |X|^2, mag = |X|, t = temporary. */
void es_features(const float Xr[ES_NBINS], const float Xi[ES_NBINS], float feat[ES_NFEAT],
                 float work[3 * ES_NBINS]) {
  float *p = work, *mag = work + ES_NBINS, *t = work + 2 * ES_NBINS;
  const int n257 = ES_NBINS;
  float tiny = 1e-8f, pw = -0.7f, logscale = 0.1f;
  DSPSplitComplex X = { (float *)Xr, (float *)Xi };       /* vDSP only reads it here */

  vDSP_zvmags(&X, 1, p, 1, ES_NBINS);                      /* p = |X|^2 */
  vvsqrtf(mag, p, &n257);                                  /* mag = |X| */
  vDSP_vsadd(mag, 1, &tiny, t, 1, ES_NBINS);               /* t = |X| + 1e-8 */
  vvlogf(t, t, &n257);                                     /* t = ln(|X| + 1e-8) */
  vDSP_vsmul(t, 1, &pw, t, 1, ES_NBINS);                   /* t = -0.7 ln(...) */
  vvexpf(t, t, &n257);                                     /* t = (|X| + 1e-8)^-0.7 */
  vDSP_vmul(Xr, 1, t, 1, feat + 0, 3, ES_NBINS);           /* Re c */
  vDSP_vmul(Xi, 1, t, 1, feat + 1, 3, ES_NBINS);           /* Im c */
  vDSP_vsadd(p, 1, &tiny, t, 1, ES_NBINS);                 /* t = |X|^2 + 1e-8 */
  vvlogf(t, t, &n257);
  vDSP_vsmul(t, 1, &logscale, feat + 2, 3, ES_NBINS);      /* 0.1 ln(|X|^2 + 1e-8) */
}

/* Steps 1 to 3 of a hop (20-engine 6.3): shift the 64 new samples into the analysis history,
   multiply by the analysis window and take the FFT. ana[511] is the newest sample. */
void es_analysis_hop(es_fft *f, const float wa[ES_NFFT], float ana[ES_NFFT], const float in[ES_HOP],
                     float frame[ES_NFFT], float Xr[ES_NBINS], float Xi[ES_NBINS]) {
  memmove(ana, ana + ES_HOP, sizeof(float) * (ES_NFFT - ES_HOP));
  memcpy(ana + ES_NFFT - ES_HOP, in, sizeof(float) * ES_HOP);
  vDSP_vmul(ana, 1, wa, 1, frame, 1, ES_NFFT);
  es_rfft(f, frame, Xr, Xi);
}

/* Streaming synthesis (20-engine 5.4): inverse FFT, multiply the last 128 samples by the synthesis
   window, add them into the overlap-add buffer, emit its first 64 samples, shift it left by 64. */
void es_synthesis_hop(es_fft *f, const float ws[ES_NFFT], const float Yr[ES_NBINS],
                      const float Yi[ES_NBINS], float frame[ES_NFFT], float ola[ES_SYNTH_LEN],
                      float out[ES_HOP]) {
  es_irfft(f, Yr, Yi, frame);
  for (int n = 0; n < ES_SYNTH_LEN; n++)
    ola[n] += frame[ES_SYNTH_START + n] * ws[ES_SYNTH_START + n];
  memcpy(out, ola, sizeof(float) * ES_HOP);
  memmove(ola, ola + ES_HOP, sizeof(float) * ES_HOP);
  memset(ola + ES_HOP, 0, sizeof(float) * ES_HOP);
}

/* Offline STFT of contract 2.1 (test-only): left-pad x with 448 zeros and take frames of 512 at hop 64,
   so frame t ends at input sample (t + 1) * 64 - 1. n must be a positive multiple of 64; T = n / 64
   frames are written to Xr, Xi as [T][257]. */
es_status es_stft_offline(const float wa[ES_NFFT], const float *x, int n, float *Xr, float *Xi) {
  if (!wa || !x || !Xr || !Xi || n <= 0 || n % ES_HOP != 0) return ES_ERR_ARG;
  es_fft f;
  if (es_fft_init(&f) != ES_OK) return ES_ERR_NOMEM;
  float frame[ES_NFFT];
  int T = n / ES_HOP;
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < ES_NFFT; i++) {
      long j = (long)ES_HOP * t + i - ES_ANA_RISE;           /* index into x (negative = padding) */
      frame[i] = (j >= 0 ? x[j] : 0.0f) * wa[i];
    }
    es_rfft(&f, frame, Xr + (size_t)t * ES_NBINS, Xi + (size_t)t * ES_NBINS);
  }
  es_fft_free(&f);
  return ES_OK;
}

/* Offline iSTFT of contract 2.1 (test-only): irfft each frame, multiply by ws and overlap-add its last
   128 samples at hop 64; frame t covers output samples 64t - 64 .. 64t + 63. y gets 64 * T samples
   aligned with the input of es_stft_offline (the last 64 samples only hold half an overlap). */
es_status es_istft_offline(const float ws[ES_NFFT], const float *Xr, const float *Xi, int T, float *y) {
  if (!ws || !Xr || !Xi || !y || T <= 0) return ES_ERR_ARG;
  es_fft f;
  if (es_fft_init(&f) != ES_OK) return ES_ERR_NOMEM;
  float frame[ES_NFFT];
  long n = (long)T * ES_HOP;
  memset(y, 0, sizeof(float) * (size_t)n);
  for (int t = 0; t < T; t++) {
    es_irfft(&f, Xr + (size_t)t * ES_NBINS, Xi + (size_t)t * ES_NBINS, frame);
    for (int i = 0; i < ES_SYNTH_LEN; i++) {
      long idx = (long)ES_HOP * t - ES_HOP + i;
      if (idx >= 0 && idx < n) y[idx] += frame[ES_SYNTH_START + i] * ws[ES_SYNTH_START + i];
    }
  }
  es_fft_free(&f);
  return ES_OK;
}
