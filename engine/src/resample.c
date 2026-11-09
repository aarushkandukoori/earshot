/* resample.c - one rational polyphase FIR resampler for both directions and both I/O rates
 * (20-engine 9, contract 4.2 "Resamplers").
 *
 * The math. To change the rate by L/M we (conceptually) put L-1 zeros between input samples, low-pass
 * filter with the prototype h at the high rate, and keep every M-th sample. Output m needs the
 * high-rate position n0 = m * M. Write n0 = L * base + p (p = n0 mod L). Only every L-th high-rate
 * sample is non-zero, so
 *     y[m] = sum_j  L * h[p + j L] * x[base - j]
 * The tables (resampler_tables.c) store taps[p][j] = L * h[p + j L], each phase reversed so that one
 * output is a single dot product over ascending memory. Moving to the next output adds M to n0:
 *     p += M;  base += p / L;  p %= L;
 * All of this is exact integer arithmetic, so the delay never drifts, whatever the block sizes.
 *
 * Decimator 48 kHz -> 16 kHz: L = 1, M = 3. Interpolator 16 -> 48 kHz: L = 3, M = 1.
 * 44.1 kHz: decimator L = 160, M = 441; interpolator L = 441, M = 160.
 *
 * Threads: es_rs_process, es_rs_need and es_rs_step_count are RT-safe (no allocation; the caller owns
 * hist). es_rs_prefill_min is non-RT (called once by es_engine_new).
 */
#include <string.h>

#include "internal.h"

int es_rs_family(int io_rate) {
  if (io_rate == 48000) return 0;
  if (io_rate == 44100) return 1;
  return -1;
}

/* hist holds Q old samples followed by up to max_in new ones. Why Q and not Q - 1: one output reads Q
   samples x[base - Q + 1 .. base]. When render feeds exactly es_rs_need(n) inputs, a call can start
   with base = consumed - 1 (the newest old sample), and then the oldest sample it reads is
   x[consumed - Q]: Q samples back. base is never further back than that (the loop runs until
   base >= end or until the n-th output, whose base is end - 1 at most), so Q old samples always suffice. */
int es_rs_hist_len(const es_rs_table *t, int max_in) {
  return t->Q + max_in;
}

void es_rs_init(es_rs *r, const es_rs_table *t, float *hist) {
  r->L = t->L;
  r->M = t->M;
  r->Q = t->Q;
  r->rev = t->rev;
  r->hist = hist;
  es_rs_reset(r);
}

void es_rs_reset(es_rs *r) {
  if (r->hist) memset(r->hist, 0, sizeof(float) * (size_t)r->Q);   /* the "past" is silence */
  r->base = 0;
  r->consumed = 0;
  r->p = 0;
}

/* Feed n_in new samples and produce up to max_out outputs. hist must have room for Q + n_in floats.
   Layout of hist during a call: hist[0 .. Q-1] = x[consumed - Q .. consumed - 1] (old samples),
   hist[Q .. Q + n_in - 1] = the new samples, so x[i] sits at hist[i - consumed + Q].
   Returns the number of outputs written. */
int es_rs_process(es_rs *r, const float *in, int n_in, float *out, int max_out) {
  if (n_in > 0) memcpy(r->hist + r->Q, in, sizeof(float) * (size_t)n_in);
  int64_t end = r->consumed + n_in;                     /* absolute index one past the newest input */
  int k = 0;
  while (k < max_out && r->base < end) {
    int64_t local = r->base - r->consumed + r->Q;       /* where x[base] sits in hist (>= Q - 1) */
    /* One output = dot product of the reversed phase with x[base - Q + 1 .. base]. */
    vDSP_dotpr(r->rev + (size_t)r->p * r->Q, 1, r->hist + local - (r->Q - 1), 1, &out[k], (vDSP_Length)r->Q);
    k++;
    r->p += r->M;                                       /* step to the next output */
    r->base += r->p / r->L;
    r->p %= r->L;
  }
  r->consumed = end;
  if (n_in > 0) memmove(r->hist, r->hist + n_in, sizeof(float) * (size_t)r->Q);   /* keep the newest Q */
  return k;
}

/* How many new inputs must be fed before n_out more outputs exist. The last of those outputs needs
   input sample base + (p + (n_out - 1) M) / L. */
int es_rs_need(const es_rs *r, int n_out) {
  if (n_out <= 0) return 0;
  int64_t last = r->base + (r->p + (int64_t)(n_out - 1) * r->M) / r->L;
  int64_t need = last + 1 - r->consumed;
  return need > 0 ? (int)need : 0;
}

/* The same stepping as es_rs_process without any samples: used to simulate the timing. */
int es_rs_step_count(es_rs *r, int n_in, int max_out) {
  int64_t end = r->consumed + n_in;
  int k = 0;
  while (k < max_out && r->base < end) {
    k++;
    r->p += r->M;
    r->base += r->p / r->L;
    r->p %= r->L;
  }
  r->consumed = end;
  return k;
}

/* P_min (20-engine 9.4): the smallest 16 kHz prefill that never starves the interpolator over 1,200
   callbacks, for an instant worker whose hops become usable on the next callback.
   The spec states it as "try P = 0, 1, 2, ... and keep the first that survives". Nothing in the
   simulation depends on P except the starvation test (avail starts at P and every later step adds
   or subtracts the same amounts), so one pass with P = 0 suffices: P_min is the largest shortfall
   need - avail seen at any callback (0 if there is none). This gives the same table, much faster. */
int es_rs_prefill_min(int io_rate, int io_frames) {
  int fam = es_rs_family(io_rate);
  if (fam < 0 || io_frames < 1 || io_frames > ES_RS_MAX_DECIM_IN) return -1;
  es_rs dec = { 0 }, itp = { 0 };
  dec.L = es_rs_decim_tables[fam].L;  dec.M = es_rs_decim_tables[fam].M;
  itp.L = es_rs_interp_tables[fam].L; itp.M = es_rs_interp_tables[fam].M;

  int64_t in_avail = 0;     /* 16 kHz input samples not yet formed into a hop */
  int64_t avail = 0;        /* 16 kHz output samples in out16 (with P = 0; may go negative) */
  int64_t pending = 0;      /* hops computed after the previous callback */
  int64_t worst = 0;        /* largest shortfall seen */
  for (int k = 0; k < 1200; k++) {
    avail += 64 * pending;                                     /* last callback's hops are usable now */
    pending = 0;
    in_avail += es_rs_step_count(&dec, io_frames, 1 << 30);   /* decimator outputs of this callback */
    int need = es_rs_need(&itp, io_frames);
    if (need - avail > worst) worst = need - avail;
    avail -= need;
    es_rs_step_count(&itp, need, io_frames);
    pending = in_avail / 64;                                   /* the worker turns them into hops */
    in_avail -= 64 * pending;
  }
  return (int)worst;
}
