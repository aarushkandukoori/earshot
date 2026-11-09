/* probe.c - the latency probe: start a run and analyze the capture with GCC-PHAT
 * (20-engine 12, contract 4 and 4.2 "Latency probe", D49).
 *
 * Who may call what (contract 4.1): es_engine_probe_start and es_engine_probe_result run on a non-RT
 * thread (the app's LatencyMeter). Call them from one thread at a time. probe_result allocates.
 *
 * How a probe works, end to end:
 *   1. es_engine_new (pipeline.c) builds one sweep: 300 Hz to 6 kHz, exponential, 0.3 s, at
 *      ES_PROBE_DBFS, with 5 ms raised-cosine edges, at the I/O rate. It also allocates the 4.5 s capture.
 *   2. es_engine_probe_start (here) zeroes the play/capture index and sets the state to RUNNING.
 *   3. Every render callback (pipeline.c, probe_step) then plays sweep k starting at 0.8 k s
 *      (k = 0..4, 4.0 s of playback) times the mute gain, and copies the mic input into the capture at the
 *      same index. So a sample played at index i that comes back through the air (or a cable) at index
 *      i + D tells us the round trip D. At 4.5 s the render sets the state to CAPTURED.
 *   4. es_engine_probe_result (here) finds D for each sweep with GCC-PHAT and returns the median.
 *
 * GCC-PHAT in plain words: correlate the capture with the sweep, but first throw away the magnitude of
 * every frequency bin and keep only its phase ("phase transform"). A pure delay D shows up as the same
 * linear phase in every bin, so after the inverse FFT all bins add up at lag D and give one sharp peak,
 * whatever the loudspeaker and mic do to the magnitudes.
 *   R = FFT(sweep), C = FFT(capture window), G = C conj(R) / (|C conj(R)| + 1e-12), g = IFFT(G)
 *   G is kept only on the bins from 300 Hz to 6 kHz, the band the sweep covers, and set to 0 elsewhere
 *   (DECISIONS L 2026-11-08). After the phase transform every bin has weight 1, so a bin where the
 *   sweep has no energy carries pure noise at full weight; dropping those bins raised psr at -36 dBFS
 *   noise (unit row probe_band_limited_psr_*).
 *   lag = argmax |g| over 0 .. 0.45 s, refined by a parabola through the peak and its two neighbors
 *   psr_db = 20 log10(|g_peak| / largest |g| in the same lag range at least 2 ms away from the peak)
 * A sweep counts as valid when psr_db >= 10 dB. ES_ERR_PROBE_WEAK only when fewer than 3 are valid.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

#define PROBE_LOG2N  16                 /* FFT size 65,536 (20-engine 12) */
#define PROBE_N      (1 << PROBE_LOG2N)
#define PROBE_SWEEPS 5
#define PROBE_PSR_OK 10.0f              /* dB: a sweep is valid at or above this */
#define PROBE_LO_HZ  300.0              /* GCC-PHAT band: the sweep's own band (DECISIONS L 2026-11-08) */
#define PROBE_HI_HZ  6000.0

/* The cached answer kept in e->probe_priv (one malloc'd block; es_engine_free frees it). */
typedef struct {
  es_status status;                     /* ES_OK or ES_ERR_PROBE_WEAK */
  es_probe_result r;
  float lag_ms[PROBE_SWEEPS];           /* per-sweep numbers, kept for debugging */
  float psr_db[PROBE_SWEEPS];
} probe_cache;

/* ------------------------------------------------------------------------------------------------
 * es_engine_probe_start: BUSY while a run is going; otherwise forget the old answer, rewind the index
 * and publish RUNNING with a release store, so the render (which loads the state with acquire) sees
 * probe_pos = 0 before it touches the capture.
 * ---------------------------------------------------------------------------------------------- */
es_status es_engine_probe_start(es_engine *e) {
  if (!e) return ES_ERR_ARG;
  if (atomic_load_explicit(&e->probe_state, memory_order_acquire) == ES_PROBE_RUNNING) return ES_ERR_BUSY;
  free(e->probe_priv);                  /* the cached result belongs to the previous run */
  e->probe_priv = NULL;
  e->probe_pos = 0;                     /* the render only reads it while RUNNING */
  atomic_store_explicit(&e->probe_state, ES_PROBE_RUNNING, memory_order_release);
  return ES_OK;
}

/* ------------------------------------------------------------------------------------------------
 * Real FFT helpers on vDSP's packed format. For a real signal of N samples, vDSP keeps N/2 complex
 * numbers: re[0] = X[0] (DC, real), im[0] = X[N/2] (Nyquist, real), and re[k] + i im[k] = X[k] for
 * k = 1 .. N/2 - 1. The other half of the spectrum is the complex conjugate, so it is not stored.
 * ---------------------------------------------------------------------------------------------- */
/* X = DFT of x[0..n) zero-padded to PROBE_N, scaled to the true DFT (vDSP's forward gives 2x). */
static void rfft_packed(FFTSetup setup, const float *x, int n, float *tmp, DSPSplitComplex *X) {
  memset(tmp, 0, sizeof(float) * PROBE_N);
  memcpy(tmp, x, sizeof(float) * (size_t)n);
  vDSP_ctoz((const DSPComplex *)tmp, 2, X, 1, PROBE_N / 2);   /* even samples -> re, odd -> im */
  vDSP_fft_zrip(setup, X, 1, PROBE_LOG2N, FFT_FORWARD);
  float half = 0.5f;
  vDSP_vsmul(X->realp, 1, &half, X->realp, 1, PROBE_N / 2);
  vDSP_vsmul(X->imagp, 1, &half, X->imagp, 1, PROBE_N / 2);
}

/* G = C conj(R) / (|C conj(R)| + 1e-12) in place in C (the PHAT weighting), on bins k_lo..k_hi only;
   every other bin is set to 0. Bin k is the frequency k * rate / PROBE_N. Bin 0 (DC) and bin N/2
   (Nyquist, kept in imagp[0]) are real numbers; they count only when the band includes them. */
static void phat(DSPSplitComplex *C, const DSPSplitComplex *R, int k_lo, int k_hi) {
  float dc = C->realp[0] * R->realp[0], ny = C->imagp[0] * R->imagp[0];
  C->realp[0] = k_lo <= 0 ? dc / (fabsf(dc) + 1e-12f) : 0.0f;
  C->imagp[0] = k_hi >= PROBE_N / 2 ? ny / (fabsf(ny) + 1e-12f) : 0.0f;
  for (int k = 1; k < PROBE_N / 2; k++) {
    if (k < k_lo || k > k_hi) {           /* outside the band: weight 0 */
      C->realp[k] = C->imagp[k] = 0.0f;
      continue;
    }
    float a = C->realp[k], b = C->imagp[k], c = R->realp[k], d = R->imagp[k];
    float re = a * c + b * d;           /* (a + ib)(c - id) */
    float im = b * c - a * d;
    float mag = sqrtf(re * re + im * im) + 1e-12f;
    C->realp[k] = re / mag;
    C->imagp[k] = im / mag;
  }
}

/* g = inverse DFT of the packed spectrum G (scaled to the true inverse: vDSP's inverse gives N x). */
static void irfft_packed(FFTSetup setup, DSPSplitComplex *G, float *g) {
  vDSP_fft_zrip(setup, G, 1, PROBE_LOG2N, FFT_INVERSE);
  vDSP_ztoc(G, 1, (DSPComplex *)g, 2, PROBE_N / 2);
  float scale = 1.0f / (float)PROBE_N;
  vDSP_vsmul(g, 1, &scale, g, 1, PROBE_N);
}

/* Peak of |g| over lags 0..max_lag, its parabolic refinement and the peak-to-sidelobe ratio.
   g is circular: index PROBE_N - 1 is lag -1, which the parabola needs when the peak is at lag 0. */
static void find_peak(const float *g, int max_lag, int excl, double *lag_out, float *psr_out) {
  int p = 0;
  float best = -1.0f;
  for (int i = 0; i <= max_lag; i++)
    if (fabsf(g[i]) > best) { best = fabsf(g[i]); p = i; }
  float ym1 = fabsf(g[p == 0 ? PROBE_N - 1 : p - 1]), y0 = best, yp1 = fabsf(g[p + 1]);
  double den = (double)ym1 - 2.0 * y0 + yp1, delta = 0.0;
  if (den < 0.0) delta = 0.5 * ((double)ym1 - yp1) / den;   /* vertex of the parabola */
  if (delta > 0.5) delta = 0.5;
  if (delta < -0.5) delta = -0.5;
  *lag_out = p + delta;
  float side = 0.0f;                                         /* largest |g| at least excl away */
  for (int i = 0; i <= max_lag; i++)
    if ((i < p - excl || i > p + excl) && fabsf(g[i]) > side) side = fabsf(g[i]);
  *psr_out = 20.0f * log10f((best + 1e-30f) / (side + 1e-30f));
}

/* Median of n values (n >= 1), sorting a small copy. Even n gives the mean of the middle two. */
static float median(const float *v, int n) {
  float s[PROBE_SWEEPS];
  memcpy(s, v, sizeof(float) * (size_t)n);
  for (int i = 1; i < n; i++)                                /* insertion sort, n <= 5 */
    for (int j = i; j > 0 && s[j - 1] > s[j]; j--) { float t = s[j]; s[j] = s[j - 1]; s[j - 1] = t; }
  return (n & 1) ? s[n / 2] : 0.5f * (s[n / 2 - 1] + s[n / 2]);
}

static float min_max_spread(const float *v, int n) {
  float lo = v[0], hi = v[0];
  for (int i = 1; i < n; i++) { if (v[i] < lo) lo = v[i]; if (v[i] > hi) hi = v[i]; }
  return hi - lo;
}

/* GCC-PHAT of each of the 5 sweeps with only the bins from lo_hz to hi_hz weighted: lag_ms[k] and
   psr_db[k] per sweep. Non-RT, allocates about 1 MB for the FFT buffers. The engine always uses
   300 Hz to 6 kHz; test_units also calls it with the full band (0 Hz to rate / 2, the E-2 behavior)
   to show what the band limit gains. ES_OK or ES_ERR_NOMEM. */
es_status es_probe_analyze_band(const es_engine *e, double lo_hz, double hi_hz, float lag_ms[5], float psr_db[5]) {
  const double sr = (double)e->cfg.io_rate;
  const int win = (int)(0.75 * sr + 0.5);                    /* capture window per sweep: 0.75 s */
  const int max_lag = (int)(0.45 * sr + 0.5);                /* lags searched: 0 .. 0.45 s */
  const int excl = (int)(0.002 * sr + 0.5);                  /* +/- 2 ms around the peak */
  const int k_lo = (int)ceil(lo_hz * PROBE_N / sr);          /* first bin at or above lo_hz */
  const int k_hi = (int)floor(hi_hz * PROBE_N / sr);         /* last bin at or below hi_hz */
  FFTSetup setup = vDSP_create_fftsetup(PROBE_LOG2N, kFFTRadix2);
  float *mem = malloc(sizeof(float) * (size_t)PROBE_N * 3);  /* tmp / g, R (re, im), C (re, im) */
  if (!setup || !mem) {
    if (setup) vDSP_destroy_fftsetup(setup);
    free(mem);
    return ES_ERR_NOMEM;
  }
  float *tmp = mem;                                          /* time-domain scratch, later g */
  DSPSplitComplex R = { mem + PROBE_N, mem + PROBE_N + PROBE_N / 2 };
  DSPSplitComplex C = { mem + 2 * PROBE_N, mem + 2 * PROBE_N + PROBE_N / 2 };

  rfft_packed(setup, e->probe_sweep, e->probe_sweep_len, tmp, &R);   /* the same R for every sweep */
  for (int k = 0; k < PROBE_SWEEPS; k++) {
    int start = k * e->probe_period;
    int len = win;
    if (start + len > e->probe_cap_len) len = e->probe_cap_len - start;   /* never at 48 or 44.1 kHz */
    rfft_packed(setup, e->probe_cap + start, len, tmp, &C);
    phat(&C, &R, k_lo, k_hi);
    irfft_packed(setup, &C, tmp);
    double lag;
    find_peak(tmp, max_lag, excl, &lag, &psr_db[k]);
    lag_ms[k] = (float)(1000.0 * lag / sr);
  }
  vDSP_destroy_fftsetup(setup);
  free(mem);
  return ES_OK;
}

/* The whole analysis of one capture: per-sweep GCC-PHAT in the sweep band, then the summary. */
static es_status analyze(const es_engine *e, probe_cache *out) {
  es_status st = es_probe_analyze_band(e, PROBE_LO_HZ, PROBE_HI_HZ, out->lag_ms, out->psr_db);
  if (st != ES_OK) return st;

  /* Valid sweeps (psr >= 10 dB): median lag, spread (max - min) and median psr. With no valid sweep
     the same numbers are taken over all 5 sweeps, so spread_ms is always filled. */
  float lag_ok[PROBE_SWEEPS], psr_ok[PROBE_SWEEPS];
  int n_ok = 0;
  for (int k = 0; k < PROBE_SWEEPS; k++)
    if (out->psr_db[k] >= PROBE_PSR_OK) { lag_ok[n_ok] = out->lag_ms[k]; psr_ok[n_ok] = out->psr_db[k]; n_ok++; }
  const float *lv = n_ok ? lag_ok : out->lag_ms, *pv = n_ok ? psr_ok : out->psr_db;
  int nv = n_ok ? n_ok : PROBE_SWEEPS;
  out->r.loop_ms = median(lv, nv);
  out->r.spread_ms = min_max_spread(lv, nv);
  out->r.psr_db = median(pv, nv);
  out->r.n_ok = n_ok;
  out->status = n_ok < 3 ? ES_ERR_PROBE_WEAK : ES_OK;
  return ES_OK;
}

/* ------------------------------------------------------------------------------------------------
 * es_engine_probe_result: ES_ERR_STATE while running or never started. After a run, the first call
 * analyzes the capture and caches the answer; later calls return the same answer until the next start.
 * Returns ES_OK, ES_ERR_PROBE_WEAK (r is still filled), ES_ERR_STATE, ES_ERR_ARG or ES_ERR_NOMEM.
 * ---------------------------------------------------------------------------------------------- */
es_status es_engine_probe_result(es_engine *e, es_probe_result *r) {
  if (!e || !r) return ES_ERR_ARG;
  memset(r, 0, sizeof(*r));
  if (atomic_load_explicit(&e->probe_state, memory_order_acquire) != ES_PROBE_CAPTURED) return ES_ERR_STATE;
  if (!e->probe_priv) {
    probe_cache *pc = calloc(1, sizeof(probe_cache));
    if (!pc) return ES_ERR_NOMEM;
    es_status st = analyze(e, pc);
    if (st != ES_OK) {
      free(pc);
      return st;
    }
    e->probe_priv = pc;
  }
  const probe_cache *pc = e->probe_priv;
  *r = pc->r;
  return pc->status;
}
