/* pipeline.c - the engine: rings, resampling, conditioning hand-off, params, render and the hop loop
 * (20-engine 8 and 11, contract 4, 4.1, 4.2 and 4.3).
 *
 * Who may call what (contract 4.1):
 *   es_engine_new / es_engine_free, es_engine_read_*            any non-RT thread
 *   es_engine_set_cond / es_engine_set_params                    ONE non-RT writer (the Swift main actor)
 *   es_engine_render                                             the audio I/O thread only (RT-safe)
 *   es_engine_run_hops                                           the worker only (RT-safe)
 *   set_mute, cue, get_stats, input_position, latency_ms         any thread (RT-safe)
 * The worker thread itself (start, stop, policy) lives in rt_thread.c; post-processing in post.c;
 * the safety-chain blocks in safety.c; the probe analysis in probe.c.
 *
 * The big picture of one audio callback (es_engine_render, 20-engine 8.3):
 *   mic (48 or 44.1 kHz) -> clamp -> decimate to 16 kHz -> in16 ring + input history
 *                        -> wake the worker
 *   out16 ring (16 kHz, filled by the worker) -> interpolate to the I/O rate -> safety chain -> speaker
 * and of one worker hop (es_engine_run_hops, 20-engine 8.4):
 *   64 samples from in16 -> each active slot's es_stream_hop -> post (gate, mix, duck, AGC, cap, room,
 *   bypass) -> 64 samples into out16.
 * out16 starts with P = P_min + slack16 zeros, so the speaker hears the mic (64 + P) / 16000 s later
 * plus the resampler filter delays: that is es_engine_latency_ms.
 *
 * Nothing here allocates after es_engine_new. Shared fields are atomics, the 4.3 hand-off, or the
 * params seqlock; each field's owner is written in struct es_engine (internal.h).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>

#include "internal.h"

/* ------------------------------------------------------------------------------------------------
 * Small helpers
 * ---------------------------------------------------------------------------------------------- */
/* 64-byte aligned, zeroed block (non-RT). */
static void *alloc_zero(size_t bytes) {
  void *p = NULL;
  if (posix_memalign(&p, 64, bytes) != 0) return NULL;
  memset(p, 0, bytes);
  return p;
}

static float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

/* Mean square of n samples as dB (silence gives -120 dB). */
static float rms_db(const float *x, int n) {
  if (n <= 0) return -120.0f;
  float s = 0.0f;
  for (int i = 0; i < n; i++) s += x[i] * x[i];
  return 10.0f * log10f(s / (float)n + 1e-12f);
}

uint64_t es_ticks_to_us(const es_engine *e, uint64_t ticks) {
  return ticks * e->tb_numer / ((uint64_t)e->tb_denom * 1000u);
}

/* ------------------------------------------------------------------------------------------------
 * Parameters: clamp, then publish through a seqlock (20-engine 8.5).
 * The writer makes seq odd, stores the 12 words, makes seq even. A reader copies the words between
 * two loads of seq; if seq was odd or changed, the copy may be torn and the reader keeps its previous
 * params (it never spins, so the audio thread and the worker never wait).
 * ---------------------------------------------------------------------------------------------- */
_Static_assert(sizeof(es_params) == ES_PARAM_WORDS * 4, "es_params must be 12 four-byte words");

static float fix(float v, float lo, float hi, float dflt) { return isfinite(v) ? clampf(v, lo, hi) : dflt; }

static void params_clamp(const es_params *in, es_params *out) {
  es_params d;
  es_params_default(&d);
  out->voice_db = fix(in->voice_db, -10.0f, 6.0f, d.voice_db);
  out->room_db = fix(in->room_db, -60.0f, -8.0f, d.room_db);
  out->max_gain_db = fix(in->max_gain_db, 0.0f, 18.0f, d.max_gain_db);
  out->out_gain_db = fix(in->out_gain_db, -40.0f, 0.0f, d.out_gain_db);
  for (int s = 0; s < ES_MAX_INSTANCES; s++) out->slot_gain[s] = fix(in->slot_gain[s], 0.0f, 1.0f, d.slot_gain[s]);
  out->xfade_ms = fix(in->xfade_ms, 4.0f, 2000.0f, d.xfade_ms);
  out->duck_own = in->duck_own != 0;
  out->gate = in->gate != 0;
  out->agc = in->agc != 0;
  out->bypass = in->bypass != 0;
  if (out->voice_db + out->max_gain_db > 24.0f) out->max_gain_db = 24.0f - out->voice_db;
}

static void params_publish(es_engine *e, const es_params *p) {
  uint32_t w[ES_PARAM_WORDS];
  memcpy(w, p, sizeof(w));
  uint32_t seq = atomic_load_explicit(&e->pseq, memory_order_relaxed);
  atomic_store_explicit(&e->pseq, seq + 1, memory_order_relaxed);        /* odd: writing */
  atomic_thread_fence(memory_order_release);                             /* odd seq before the words */
  for (int i = 0; i < ES_PARAM_WORDS; i++) atomic_store_explicit(&e->pwords[i], w[i], memory_order_relaxed);
  atomic_store_explicit(&e->pseq, seq + 2, memory_order_release);        /* even: done */
}

/* RT-safe reader. Returns 1 if *p was updated, 0 if it kept the previous copy (torn read). */
static int params_read(es_engine *e, es_params *p) {
  uint32_t s1 = atomic_load_explicit(&e->pseq, memory_order_acquire);
  if (s1 & 1u) return 0;
  uint32_t w[ES_PARAM_WORDS];
  for (int i = 0; i < ES_PARAM_WORDS; i++) w[i] = atomic_load_explicit(&e->pwords[i], memory_order_relaxed);
  atomic_thread_fence(memory_order_acquire);                             /* the words before seq again */
  uint32_t s2 = atomic_load_explicit(&e->pseq, memory_order_relaxed);
  if (s1 != s2) return 0;
  memcpy(p, w, sizeof(w));
  return 1;
}

void es_engine_set_params(es_engine *e, const es_params *p) {
  if (!e || !p) return;
  es_params c;
  params_clamp(p, &c);
  params_publish(e, &c);
}

/* ------------------------------------------------------------------------------------------------
 * The probe sweep (20-engine 12): 300 Hz to 6 kHz exponential sweep in 0.3 s at ES_PROBE_DBFS with
 * 5 ms raised-cosine edges, built once here at the I/O rate (double math, float store).
 * ---------------------------------------------------------------------------------------------- */
static void make_sweep(float *dst, int n, double sr) {
  const double f1 = 300.0, f2 = 6000.0, T = 0.3, k = log(f2 / f1);
  const double A = pow(10.0, ES_PROBE_DBFS / 20.0);
  int edge = (int)(0.005 * sr + 0.5);
  for (int i = 0; i < n; i++) {
    double t = i / sr;
    double x = A * sin(2.0 * ES_PI * f1 * T / k * (exp(t * k / T) - 1.0));
    if (i < edge) x *= 0.5 - 0.5 * cos(ES_PI * i / edge);
    else if (i >= n - edge) x *= 0.5 - 0.5 * cos(ES_PI * (n - 1 - i) / edge);
    dst[i] = (float)x;
  }
}

/* ------------------------------------------------------------------------------------------------
 * es_engine_new (non-RT, 20-engine 8.2) and es_engine_free.
 * ---------------------------------------------------------------------------------------------- */
es_status es_engine_new(const es_model *m, const es_engine_config *cfg, es_engine **out) {
  if (out) *out = NULL;
  if (!m || !cfg || !out) return ES_ERR_ARG;
  /* 1. Validate. */
  int fam = es_rs_family(cfg->io_rate);
  if (fam < 0) return ES_ERR_RATE;
  if (cfg->io_frames < 16 || cfg->io_frames > cfg->max_frames || cfg->max_frames > 4096) return ES_ERR_ARG;
  if (cfg->n_instances < 1 || cfg->n_instances > ES_MAX_INSTANCES) return ES_ERR_ARG;
  if (cfg->slack16 != 22 && cfg->slack16 != 43 && cfg->slack16 != 64) return ES_ERR_ARG;

  /* 2. Allocate and zero everything. On any failure es_engine_free cleans up what exists. */
  es_engine *e = alloc_zero(sizeof(es_engine));
  if (!e) return ES_ERR_NOMEM;
  e->model = m;
  e->cfg = *cfg;
  e->fam = fam;
  e->n_inst = cfg->n_instances;
  int ok = 1;
  for (int s = 0; s < e->n_inst; s++) {
    e->st[s] = es_stream_new(m);
    e->cond[s][0] = es_cond_new();                 /* n_tokens = 0: every slot starts empty (4.3) */
    e->cond[s][1] = es_cond_new();
    e->outhist_buf[s] = alloc_zero(sizeof(float) * ES_OUTHIST_SIZE);
    ok = ok && e->st[s] && e->cond[s][0] && e->cond[s][1] && e->outhist_buf[s];
  }
  const es_rs_table *dt = &es_rs_decim_tables[fam], *it = &es_rs_interp_tables[fam];
  e->in16_buf = alloc_zero(sizeof(float) * ES_IN16_SIZE);
  e->out16_buf = alloc_zero(sizeof(float) * ES_OUT16_SIZE);
  e->inhist_buf = alloc_zero(sizeof(float) * ES_HISTORY_SAMPLES);
  e->timings_buf = alloc_zero(sizeof(uint16_t) * ES_TIMINGS_SIZE);
  e->dec_hist = alloc_zero(sizeof(float) * (size_t)es_rs_hist_len(dt, ES_RS_MAX_DECIM_IN));
  e->itp_hist = alloc_zero(sizeof(float) * (size_t)es_rs_hist_len(it, ES_RS_MAX_INTERP_IN));
  e->xin = alloc_zero(sizeof(float) * 4096);
  e->d16 = alloc_zero(sizeof(float) * ES_RS_MAX_INTERP_IN);
  e->u16 = alloc_zero(sizeof(float) * ES_RS_MAX_INTERP_IN);
  e->y48 = alloc_zero(sizeof(float) * 4096);
  e->g48 = alloc_zero(sizeof(float) * 4096);
  e->probe_sweep_len = (int)(0.3 * cfg->io_rate + 0.5);
  e->probe_period = (int)(0.8 * cfg->io_rate + 0.5);
  e->probe_play_len = 5 * e->probe_period;          /* 4.0 s */
  e->probe_cap_len = (int)(4.5 * cfg->io_rate + 0.5);
  e->probe_sweep = alloc_zero(sizeof(float) * (size_t)e->probe_sweep_len);
  e->probe_cap = alloc_zero(sizeof(float) * (size_t)e->probe_cap_len);
  ok = ok && e->in16_buf && e->out16_buf && e->inhist_buf && e->timings_buf && e->dec_hist && e->itp_hist &&
       e->xin && e->d16 && e->u16 && e->y48 && e->g48 && e->probe_sweep && e->probe_cap;
  if (ok) ok = semaphore_create(mach_task_self(), &e->sem, SYNC_POLICY_FIFO, 0) == KERN_SUCCESS;
  if (ok) e->sem_made = 1;
  if (!ok) {
    es_engine_free(e);
    return ES_ERR_NOMEM;
  }
  ring_init(&e->in16, e->in16_buf, ES_IN16_SIZE);
  ring_init(&e->out16, e->out16_buf, ES_OUT16_SIZE);
  hist_init(&e->inhist, e->inhist_buf, ES_HISTORY_SAMPLES);
  for (int s = 0; s < e->n_inst; s++) hist_init(&e->outhist[s], e->outhist_buf[s], ES_OUTHIST_SIZE);
  ring16_init(&e->timings, e->timings_buf, ES_TIMINGS_SIZE);
  es_rs_init(&e->dec, dt, e->dec_hist);
  es_rs_init(&e->itp, it, e->itp_hist);
  make_sweep(e->probe_sweep, e->probe_sweep_len, (double)cfg->io_rate);

  /* Timebase and the worker's time-constraint policy (contract 4.2, 20-engine 8.4):
     period = max(4 ms, io_frames / io_rate), computation = min(period / 2, ceil(period / 4 ms) x 1 ms),
     constraint = 0.625 x period. Stored in Mach ticks (ns * denom / numer). */
  mach_timebase_info_data_t tb;
  mach_timebase_info(&tb);
  e->tb_numer = tb.numer;
  e->tb_denom = tb.denom;
  double period = fmax(0.004, (double)cfg->io_frames / cfg->io_rate);
  double computation = fmin(period / 2.0, ceil(period / 0.004 - 1e-9) * 0.001);
  double constraint = 0.625 * period;
  e->pol_period = (uint64_t)(period * 1e9 * tb.denom / tb.numer);
  e->pol_computation = (uint64_t)(computation * 1e9 * tb.denom / tb.numer);
  e->pol_constraint = (uint64_t)(constraint * 1e9 * tb.denom / tb.numer);

  /* 3. and 4. P_min (9.4) and the prefill of P = P_min + slack16 zeros. */
  e->p_min = es_rs_prefill_min(cfg->io_rate, cfg->io_frames);
  e->slack16 = cfg->slack16;
  for (int left = e->p_min + e->slack16; left > 0;) {   /* u16 is all zeros here; at most 1,613 */
    int k = left < ES_RS_MAX_INTERP_IN ? left : ES_RS_MAX_INTERP_IN;
    ring_write(&e->out16, e->u16, (uint32_t)k);
    left -= k;
  }
  e->S = e->ins = e->din = e->ins_pending = e->din_pending = 0;
  e->last_underrun = -1;

  /* 5. Safety state: muted (the first set_mute(e, 0) arms the 400 ms fade-in), default params. */
  es_params d;
  es_params_default(&d);
  params_publish(e, &d);
  e->wp = d;
  e->rp = d;
  es_post_init(&e->post, &d);
  float sr = (float)cfg->io_rate;
  e->ramp1ms = (int)ceilf(sr / 1000.0f);
  es_ramp_init(&e->conceal, sr, 1.0f);
  es_ramp_init(&e->mute, sr, 0.0f);
  e->mute_applied = 1;
  atomic_store_explicit(&e->mute_req, 1, memory_order_relaxed);
  e->trim_target = powf(10.0f, d.out_gain_db / 20.0f);
  es_ramp_init(&e->trim, sr, e->trim_target);
  es_limiter_init(&e->lim, sr, -3.0f, 50.0f);
  es_guard_init(&e->guard, sr);
  es_cue_init(&e->cue, sr);
  atomic_store_explicit(&e->s_slack16, cfg->slack16, memory_order_relaxed);
  for (int s = 0; s < ES_MAX_INSTANCES; s++) {
    atomic_store_explicit(&e->active[s], 0, memory_order_relaxed);
    atomic_store_explicit(&e->ack[s], 0, memory_order_relaxed);
  }
  atomic_thread_fence(memory_order_release);        /* everything above is visible to the threads */
  *out = e;
  return ES_OK;
}

void es_engine_free(es_engine *e) {
  if (!e) return;
  es_engine_stop_worker(e);                          /* a no-op when no worker runs */
  for (int s = 0; s < ES_MAX_INSTANCES; s++) {
    es_stream_free(e->st[s]);
    es_cond_free(e->cond[s][0]);
    es_cond_free(e->cond[s][1]);
    free(e->outhist_buf[s]);
  }
  free(e->in16_buf);
  free(e->out16_buf);
  free(e->inhist_buf);
  free(e->timings_buf);
  free(e->dec_hist);
  free(e->itp_hist);
  free(e->xin);
  free(e->d16);
  free(e->u16);
  free(e->y48);
  free(e->g48);
  free(e->probe_sweep);
  free(e->probe_cap);
  free(e->probe_priv);
  if (e->sem_made) semaphore_destroy(mach_task_self(), e->sem);
  free(e);
}

void es_engine_set_manual_worker(es_engine *e, int on) {
  if (!e) return;
  atomic_store_explicit(&e->manual_worker, on ? 1 : 0, memory_order_release);
  if (!on && !atomic_load_explicit(&e->worker_running, memory_order_acquire))
    for (int s = 0; s < e->n_inst; s++)              /* nobody runs hops: changes count as picked up */
      atomic_store_explicit(&e->ack[s], atomic_load_explicit(&e->active[s], memory_order_acquire),
                            memory_order_release);
}

/* "A worker runs hops": the real thread, or a test or tool that calls es_engine_run_hops itself. */
static int hops_running(es_engine *e) {
  return atomic_load_explicit(&e->worker_running, memory_order_acquire) ||
         atomic_load_explicit(&e->manual_worker, memory_order_acquire);
}

/* ------------------------------------------------------------------------------------------------
 * Conditioning hand-off (contract 4.3, 20-engine 8.5). One writer thread.
 *   BUSY while active != ack (the worker has not picked up the last change yet).
 *   Otherwise copy into the buffer the worker is NOT using, set its reset flag, then flip `active`
 *   with a release store; the worker's acquire load of `active` then also sees the copied data.
 * ---------------------------------------------------------------------------------------------- */
es_status es_engine_set_cond(es_engine *e, int slot, const es_cond *c, int reset_state) {
  if (!e || slot < 0 || slot >= e->n_inst) return ES_ERR_ARG;
  if (c && (c->n_tokens < 4 || c->n_tokens > ES_MAX_TOKENS)) return ES_ERR_ARG;
  int a = atomic_load_explicit(&e->active[slot], memory_order_relaxed);   /* only this thread stores it */
  int k = atomic_load_explicit(&e->ack[slot], memory_order_acquire);
  if (a != k) return ES_ERR_BUSY;
  int b = 1 - a;
  if (c) memcpy(e->cond[slot][b], c, sizeof(es_cond));                 /* 51 KB */
  else e->cond[slot][b]->n_tokens = 0;                                  /* empty slot */
  /* A slot waking up from empty always resets (its stream state is stale). A reset the active buffer
     still carries (asked for while no worker ran, so never applied) moves to the new buffer as well,
     so a second set_cond before the worker restarts cannot lose it. With a running worker that flag
     is already 0 here: the worker clears it before it acknowledges the buffer (release), and we
     read ack (acquire) above. The flags are atomics because a worker that is just starting may clear
     a flag while this thread reads it; either value is fine (at worst one extra reset). */
  int carry = atomic_load_explicit(&e->reset_pending[slot][a], memory_order_relaxed);
  atomic_store_explicit(&e->reset_pending[slot][b], reset_state || e->cond[slot][a]->n_tokens == 0 || carry,
                        memory_order_relaxed);
  atomic_store_explicit(&e->active[slot], b, memory_order_release);
  if (!hops_running(e)) atomic_store_explicit(&e->ack[slot], b, memory_order_release);
  return ES_OK;
}

/* Mute and cue: one atomic int each; the render picks them up at its next callback. */
void es_engine_set_mute(es_engine *e, int mute) {
  if (e) atomic_store_explicit(&e->mute_req, mute ? 1 : 0, memory_order_release);
}

void es_engine_cue(es_engine *e, int beeps) {
  if (!e) return;
  if (beeps < 1) beeps = 1;
  if (beeps > 3) beeps = 3;
  atomic_store_explicit(&e->cue_req, beeps, memory_order_release);
}

/* ------------------------------------------------------------------------------------------------
 * Render (audio thread, 20-engine 8.3). Helpers first.
 * ---------------------------------------------------------------------------------------------- */
/* Step 3: fill u16[0..need) from out16 with the one accounting rule. The out16 read count R should
   equal want = S - ins - din (S = every 16 kHz sample the interpolator consumed, real or zero).
     R > want: planned zeros first (a slack step or input dropped by a full in16 is paid here);
     R < want: skip samples whose time has passed (they arrived after an underrun);
   then copy real samples; whatever is still missing is unplanned zeros (an underrun).
   Writes how many planned zeros (zp), real samples (nr) and unplanned zeros (zu) u16 holds. */
static void pull_out16(es_engine *e, int need, int *zp, int *nr, int *zu) {
  int64_t R = (int64_t)ring_total_read(&e->out16);
  int64_t want = e->S - e->ins - e->din;
  int pos = 0;
  if (R > want) {
    int64_t z = R - want;
    if (z > need) z = need;
    memset(e->u16, 0, sizeof(float) * (size_t)z);
    pos = (int)z;
  } else if (R < want) {
    int64_t sk = want - R, have = (int64_t)ring_count(&e->out16);
    if (sk > have) sk = have;
    ring_skip(&e->out16, (uint32_t)sk);
  }
  *zp = pos;
  int r = need - pos;
  uint32_t have = ring_count(&e->out16);
  if ((uint32_t)r > have) r = (int)have;
  ring_read(&e->out16, e->u16 + pos, (uint32_t)r);
  pos += r;
  *nr = r;
  *zu = need - pos;
  memset(e->u16 + pos, 0, sizeof(float) * (size_t)(need - pos));
  e->S += need;
}

/* Output index (0..n) whose source is 16 kHz input k16 of this callback (n outputs from need inputs). */
static int out_index(int k16, int need, int n) {
  if (need <= 0) return n;
  int64_t v = ((int64_t)k16 * n + need / 2) / need;
  return v > n ? n : (int)v;
}

/* Step 5: the conceal ramp (1 ms linear) over y[0..n).
   - planned zeros at the start (zp > 0): the gain is already 0 there (they were applied only after
     the ramp reached 0); it rises again over 1 ms from the first output fed by real samples;
   - unplanned zeros at the end (zu > 0): the ramp down is placed so the gain reaches 0 where they
     start (as far as the callback allows);
   - planned zeros still pending: ramp down over the last 1 ms of this callback (or from its start if
     it is shorter); they are applied at the start of the callback after the gain reached 0.
   The ramp down starts one sample early so float rounding in the phase can never leave it short. */
static void conceal_apply(es_engine *e, float *y, int n, int need, int zp, int nr, int zu, int pending) {
  es_ramp *r = &e->conceal;
  int a = zp > 0 ? out_index(zp, need, n) : 0;     /* first output fed by real samples */
  es_ramp_process(r, y, a);
  if (pending || zu > 0) {
    int b = zu > 0 ? out_index(zp + nr, need, n) : n;   /* where the gain must be 0 */
    if (b < a) b = a;
    int s = b - e->ramp1ms - 1;                          /* where the ramp down starts */
    if (s < a) s = a;
    if (r->target != 0.0f) {
      es_ramp_process(r, y + a, s - a);
      es_ramp_to(r, 0.0f, 1.0f, 0);
      es_ramp_process(r, y + s, n - s);
    } else {
      es_ramp_process(r, y + a, n - a);
    }
  } else {
    if (nr > 0 && r->target != 1.0f) es_ramp_to(r, 1.0f, 1.0f, 0);   /* real samples flow again */
    es_ramp_process(r, y + a, n - a);
  }
}

/* Step 8: while a probe runs, the output is the sweep schedule times the mute gain, and the mic
   input is captured at the same index (20-engine 12). */
static void probe_step(es_engine *e, const float *xin, float *out, const float *g, int n) {
  if (atomic_load_explicit(&e->probe_state, memory_order_acquire) != ES_PROBE_RUNNING) return;
  for (int i = 0; i < n; i++) {
    int pos = e->probe_pos + i;
    if (pos < e->probe_cap_len) e->probe_cap[pos] = xin[i];
    float sw = 0.0f;
    if (pos < e->probe_play_len) {
      int t = pos % e->probe_period;                 /* sweep k starts at 0.8 k s */
      if (t < e->probe_sweep_len) sw = e->probe_sweep[t];
    }
    out[i] = sw * g[i];
  }
  e->probe_pos += n;
  if (e->probe_pos >= e->probe_cap_len) {
    e->probe_pos = e->probe_cap_len;
    uint64_t runs = atomic_load_explicit(&e->s_probe_runs, memory_order_relaxed);
    atomic_store_explicit(&e->s_probe_runs, runs + 1, memory_order_relaxed);
    atomic_store_explicit(&e->probe_state, ES_PROBE_CAPTURED, memory_order_release);
  }
}

/* Step 6: count an underrun and decide on a slack step (two underruns within 60 s, slack < 64). */
static void note_underrun(es_engine *e) {
  uint64_t u = atomic_load_explicit(&e->s_underruns, memory_order_relaxed);
  atomic_store_explicit(&e->s_underruns, u + 1, memory_order_relaxed);
  int64_t now = (int64_t)e->io_pos;
  if (e->last_underrun >= 0 && now - e->last_underrun <= 60LL * e->cfg.io_rate &&
      e->slack16 < 64 && e->grow_pending == 0) {
    e->grow_pending = ES_SLACK_STEP;
    e->last_underrun = -1;                           /* the next step needs two new underruns */
  } else {
    e->last_underrun = now;
  }
}

/* One chunk of at most max_frames samples: steps 1 to 8 of 20-engine 8.3.
   Note for the app: if render runs before the worker thread starts, in16 fills up (4,096 samples) and
   the worker's first wake then runs up to 64 stale hops in one go; their outputs are skipped as late.
   Start the worker before RemoteIO starts calling render. */
static void render_chunk(es_engine *e, const float *in, float *out, int n) {
  /* Planned zeros wait for the conceal gain to reach 0 (step 5); apply them now if it has. */
  if ((e->ins_pending || e->din_pending) && e->conceal.gain == 0.0f && e->conceal.target == 0.0f) {
    e->ins += e->ins_pending;
    e->din += e->din_pending;
    e->ins_pending = e->din_pending = 0;
  }

  /* 1. Input: clamp, decimate, push to in16 (what does not fit is dropped and paid back as planned
        zeros) and to the input history. */
  for (int i = 0; i < n; i++) {
    float v = in ? in[i] : 0.0f;
    e->xin[i] = isfinite(v) ? clampf(v, -1.0f, 1.0f) : 0.0f;
  }
  int nd = es_rs_process(&e->dec, e->xin, n, e->d16, ES_RS_MAX_INTERP_IN);
  uint32_t space = ring_space(&e->in16);
  uint32_t wr = (uint32_t)nd < space ? (uint32_t)nd : space;
  ring_write(&e->in16, e->d16, wr);
  int dropped = wr < (uint32_t)nd;
  if (dropped) e->din_pending += nd - (int)wr;
  hist_write(&e->inhist, e->d16, (uint32_t)nd);

  /* 2. Wake the worker thread. */
  if (atomic_load_explicit(&e->worker_running, memory_order_acquire) && ring_count(&e->in16) >= ES_HOP)
    semaphore_signal(e->sem);

  /* 3. Pull 16 kHz output, 4. interpolate to exactly n samples. */
  int need = es_rs_need(&e->itp, n), zp, nr, zu;
  pull_out16(e, need, &zp, &nr, &zu);
  es_rs_process(&e->itp, e->u16, need, e->y48, n);

  /* 6. Underruns (only while hops run) and slack growth. Done before step 5 (the conceal ramp, which
        is stage 1 of the safety chain below) so a new slack step starts its ramp down this callback. */
  if ((zu > 0 || dropped) && hops_running(e) && !e->cb_underrun) {
    note_underrun(e);
    e->cb_underrun = 1;                              /* at most one count per callback (8.3 step 3) */
  }
  if (e->grow_pending > 0 && e->slack16 + e->grow_pending > 64) e->grow_pending = 0;   /* never past 64 */
  if (e->grow_pending > 0) {
    int quiet = 1;                                   /* every active slot's p_target < 0.2 */
    for (int s = 0; s < e->n_inst; s++)
      if (atomic_load_explicit(&e->s_p_target[s], memory_order_relaxed) >= 0.2f) quiet = 0;
    if (quiet) {
      e->ins_pending += e->grow_pending;
      e->slack16 += e->grow_pending;
      e->grow_pending = 0;
      atomic_store_explicit(&e->s_slack16, e->slack16, memory_order_relaxed);
    }
  }

  /* 7. Safety chain (20-engine 11). */
  float *y = e->y48;
  for (int i = 0; i < n; i++) {                      /* stage 0: sanitize */
    float v = y[i];
    y[i] = isfinite(v) ? clampf(v, -1000.0f, 1000.0f) : 0.0f;
  }
  conceal_apply(e, y, n, need, zp, nr, zu, e->ins_pending > 0 || e->din_pending > 0);   /* stage 1 */
  for (int i = 0; i < n; i++) e->g48[i] = 1.0f;      /* stage 2: mute ramp gains, kept for the probe */
  es_ramp_process(&e->mute, e->g48, n);
  for (int i = 0; i < n; i++) y[i] *= e->g48[i];
  es_ramp_process(&e->trim, y, n);                   /* stage 3: output trim */
  int howl = es_guard_process(&e->guard, e->xin, y, n);   /* stage 4: runaway guard */
  es_cue_add(&e->cue, y, n);                         /* stage 5: cue beeps (heard while muted) */
  float gmin = es_limiter_process_min(&e->lim, y, n);    /* stage 6: -3 dBFS limiter */
  es_softclip(y, n);                                 /* stage 7 */
  for (int i = 0; i < n; i++) y[i] = fminf(fmaxf(y[i], -1.0f), 1.0f);   /* stage 8: hard clip */
  memcpy(out, y, sizeof(float) * (size_t)n);

  /* 8. Latency probe. */
  probe_step(e, e->xin, out, e->g48, n);

  /* Meters. */
  atomic_store_explicit(&e->s_in_rms_db, rms_db(e->xin, n), memory_order_relaxed);
  atomic_store_explicit(&e->s_out_rms_db, rms_db(out, n), memory_order_relaxed);
  atomic_store_explicit(&e->s_limiter_gr_db, -20.0f * log10f(gmin), memory_order_relaxed);
  atomic_store_explicit(&e->s_howl, howl, memory_order_relaxed);
  e->io_pos += (uint64_t)n;
}

void es_engine_render(es_engine *e, const float *in, float *out, int n) {
  if (!e || !out || n <= 0) return;
  /* Once per callback: params (seqlock), mute and cue requests. */
  params_read(e, &e->rp);
  float trim = powf(10.0f, e->rp.out_gain_db / 20.0f);
  if (trim != e->trim_target) {
    e->trim_target = trim;
    es_ramp_to(&e->trim, trim, 20.0f, 0);            /* 20 ms linear */
  }
  int mute = atomic_load_explicit(&e->mute_req, memory_order_acquire);
  if (mute != e->mute_applied) {
    if (mute) es_ramp_to(&e->mute, 0.0f, 5.0f, 0);   /* 5 ms linear down */
    else es_ramp_to(&e->mute, 1.0f, 400.0f, 1);      /* every unmute: 400 ms raised cosine */
    e->mute_applied = mute;
  }
  int beeps = atomic_exchange_explicit(&e->cue_req, 0, memory_order_acq_rel);
  if (beeps) es_cue_start(&e->cue, beeps);
  atomic_store_explicit(&e->s_last_frames, n, memory_order_relaxed);

  /* Callbacks larger than max_frames run in chunks of at most max_frames. One starved callback
     counts one underrun however many chunks it has (cb_underrun). */
  e->cb_underrun = 0;
  int mf = e->cfg.max_frames;
  for (int off = 0; off < n; off += mf) {
    int c = n - off < mf ? n - off : mf;
    render_chunk(e, in ? in + off : NULL, out + off, c);
  }
}

/* ------------------------------------------------------------------------------------------------
 * The worker body (20-engine 8.4): run every complete hop waiting in in16, at most 64 per call.
 * ---------------------------------------------------------------------------------------------- */
int es_engine_run_hops(es_engine *e) {
  if (!e) return 0;
  int hops = 0;
  while (hops < ES_MAX_HOPS_CALL && ring_count(&e->in16) >= ES_HOP) {
    uint64_t t0 = mach_absolute_time();
    ring_read(&e->in16, e->x, ES_HOP);
    params_read(e, &e->wp);                                      /* keeps the previous on a torn read */
    memcpy(e->xd, e->dl, sizeof(e->xd));                         /* xd = input of the previous hop (8.6) */
    memcpy(e->dl, e->x, sizeof(e->dl));
    uint64_t nan = 0;
    for (int s = 0; s < e->n_inst; s++) {
      /* Conditioning hand-off (4.3): use the buffer `active` points at, reset first if that buffer
         asks for it, then acknowledge so the writer may prepare the other buffer. */
      int a = atomic_load_explicit(&e->active[s], memory_order_acquire);
      const es_cond *c = e->cond[s][a];
      if (a != e->last_a[s]) {
        e->n_switches[s]++;
        e->last_a[s] = a;
      }
      if (atomic_load_explicit(&e->reset_pending[s][a], memory_order_relaxed)) {
        es_stream_reset(e->st[s]);
        atomic_store_explicit(&e->reset_pending[s][a], 0, memory_order_relaxed);
        e->n_resets[s]++;
      }
      atomic_store_explicit(&e->ack[s], a, memory_order_release);
      e->act[s] = c->n_tokens >= 4 && c->n_tokens <= ES_MAX_TOKENS;   /* active slot */
      if (e->act[s]) {
        nan += (uint64_t)es_stream_hop(e->st[s], c, e->x, e->ys[s], e->hd[s]);
      } else {                                                   /* empty slot: zeros, no compute */
        memset(e->ys[s], 0, sizeof(e->ys[s]));
        e->hd[s][0] = e->hd[s][1] = 0.0f;
      }
      hist_write(&e->outhist[s], e->ys[s], ES_HOP);              /* raw output, before gate and gains */
    }
    float pt_max, ps_max;
    es_post_hop(&e->post, &e->wp, e->n_inst, (const float (*)[ES_HOP])e->ys, (const float (*)[2])e->hd,
                e->act, e->xd, e->y, &pt_max, &ps_max);
    if (ring_space(&e->out16) >= ES_HOP) ring_write(&e->out16, e->y, ES_HOP);
    uint64_t us = es_ticks_to_us(e, mach_absolute_time() - t0);
    if (ring16_space(&e->timings) >= 1) ring16_write1(&e->timings, us > 65535 ? 65535 : (uint16_t)us);

    /* Stats (this thread is the only writer of these fields, so load + store is enough). */
    atomic_store_explicit(&e->s_hops, atomic_load_explicit(&e->s_hops, memory_order_relaxed) + 1,
                          memory_order_relaxed);
    if (nan)
      atomic_store_explicit(&e->s_nan_resets, atomic_load_explicit(&e->s_nan_resets, memory_order_relaxed) + nan,
                            memory_order_relaxed);
    for (int s = 0; s < e->n_inst; s++)
      atomic_store_explicit(&e->s_p_target[s], e->act[s] ? e->hd[s][0] : 0.0f, memory_order_relaxed);
    atomic_store_explicit(&e->s_p_self, ps_max, memory_order_relaxed);
    atomic_store_explicit(&e->s_agc_gain_db, e->post.agc_db, memory_order_relaxed);
    atomic_store_explicit(&e->s_compute_us_last, (float)us, memory_order_relaxed);
    if ((float)us > atomic_load_explicit(&e->s_compute_us_max, memory_order_relaxed))
      atomic_store_explicit(&e->s_compute_us_max, (float)us, memory_order_relaxed);
    hops++;
  }
  return hops;
}

/* ------------------------------------------------------------------------------------------------
 * Stats, latency and reads (20-engine 8.7).
 * ---------------------------------------------------------------------------------------------- */
void es_engine_get_stats(const es_engine *ce, es_stats *s) {
  if (!ce || !s) return;
  es_engine *e = (es_engine *)ce;                    /* atomic loads take a non-const pointer */
  s->hops = atomic_load_explicit(&e->s_hops, memory_order_relaxed);
  s->underruns = atomic_load_explicit(&e->s_underruns, memory_order_relaxed);
  s->nan_resets = atomic_load_explicit(&e->s_nan_resets, memory_order_relaxed);
  s->probe_runs = atomic_load_explicit(&e->s_probe_runs, memory_order_relaxed);
  s->slack16 = atomic_load_explicit(&e->s_slack16, memory_order_relaxed);
  s->last_frames = atomic_load_explicit(&e->s_last_frames, memory_order_relaxed);
  s->howl = atomic_load_explicit(&e->s_howl, memory_order_relaxed);
  for (int i = 0; i < ES_MAX_INSTANCES; i++)
    s->p_target[i] = atomic_load_explicit(&e->s_p_target[i], memory_order_relaxed);
  s->p_self = atomic_load_explicit(&e->s_p_self, memory_order_relaxed);
  s->in_rms_db = atomic_load_explicit(&e->s_in_rms_db, memory_order_relaxed);
  s->out_rms_db = atomic_load_explicit(&e->s_out_rms_db, memory_order_relaxed);
  s->agc_gain_db = atomic_load_explicit(&e->s_agc_gain_db, memory_order_relaxed);
  s->limiter_gr_db = atomic_load_explicit(&e->s_limiter_gr_db, memory_order_relaxed);
  s->compute_us_last = atomic_load_explicit(&e->s_compute_us_last, memory_order_relaxed);
  s->compute_us_max = atomic_load_explicit(&e->s_compute_us_max, memory_order_relaxed);
}

/* Earshot's delay over a pass-through: (64 + P_min + slack16) / 16 kHz plus both resampler delays. */
float es_engine_latency_ms(const es_engine *ce) {
  if (!ce) return 0.0f;
  es_engine *e = (es_engine *)ce;
  int slack = atomic_load_explicit(&e->s_slack16, memory_order_relaxed);
  return 1000.0f * (float)(ES_HOP + e->p_min + slack) / (float)ES_SR + ES_RS_DELAY_MS[e->fam];
}

uint64_t es_engine_input_position(const es_engine *ce) {
  if (!ce) return 0;
  return hist_position((es_hist *)&ce->inhist);
}

int es_engine_read_input(es_engine *e, uint64_t *cursor, float *dst, int max) {
  if (!e || !cursor || !dst || max <= 0) return 0;
  return hist_read(&e->inhist, cursor, dst, max);
}

int es_engine_read_output(es_engine *e, int slot, float *dst, int n) {
  if (!e || !dst || n <= 0 || slot < 0 || slot >= e->n_inst) return 0;
  if (n > ES_OUTHIST_READ) n = ES_OUTHIST_READ;
  return hist_read_last(&e->outhist[slot], dst, n);
}

int es_engine_read_timings(es_engine *e, uint16_t *us, int max) {
  if (!e || !us || max <= 0) return 0;
  return (int)ring16_read(&e->timings, us, (uint32_t)max);
}
