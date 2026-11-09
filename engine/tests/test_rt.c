/* test_rt.c - the real-time rules under load (20-engine 14.3, contract 4.1): 0 allocations, 0 frees
 * and 0 lock calls on the audio thread and the worker thread over RT_HOPS hops with 3 instances.
 *
 * Run it through the Makefile, which loads the counter:
 *     DYLD_INSERT_LIBRARIES=build/rt_guard.dylib RT_HOPS=10000 build/test_rt        (make test)
 *     make rt                                       (100,000 hops, under nice -n 15)
 *     build/test_rt --export ../ml/exports/random_v1    real weights instead of the synthetic model
 *
 * Threads in this test
 *   main     creates everything (non-RT), starts the engine's real worker (rt_thread.c, which runs
 *            es_engine_run_hops each time render signals it), then waits.
 *   audio    plays the RemoteIO callback: es_engine_render with sizes drawn from {16, 64, 128, 480,
 *            1104}, plus set_mute, cue, get_stats, input_position and a direct es_stream_hop on its
 *            own stream. After each callback it waits (sched_yield) until the worker caught up, so the
 *            test is paced by the worker like the real app.
 *   control  plays the app's control queue during the measured window (E-2 review): every ~1 ms it
 *            swaps conditioning (warm, cold with reset, and NULL = empty slot), changes the params
 *            (seqlock; bypass, room, AGC and slot gains toggle) and starts latency probes, so the
 *            worker's reset and buffer-flip paths and render's probe path run under the guard too.
 *            This thread is not guarded (it is allowed to allocate) and is the only set_cond caller.
 * Both the audio thread and the worker are guarded. The guard counts allocations, frees, locks and
 * waits (rt_guard.c); the audio thread must make 0 of each, the worker 0 allocations, frees and locks
 * (its only wait is the idle semaphore_wait between callbacks: at most one per wake-up, checked since
 * E-3 against the engine's worker_wakes counter). The first 100 hops
 * are a warm-up counted on their own (20-engine 14.3: setup inside Accelerate is allowed only at
 * create time; a warm-up allocation means a call should move into _new or the worker's setup). It is
 * 0 today, so it is checked too.
 * Control runs. 2026-11-07: without the worker's BLASSetThreading + warm-up call in rt_thread.c the
 * warm-up shows exactly 1 allocation (the first BLAS call on the worker thread). 2026-11-08 (fix
 * pass, scratch copy): a malloc_type_malloc planted in run_hops and an os_unfair_lock_lock_with_options
 * planted in render both fail rt_allocations / rt_lock_calls with the widened guard.
 */
#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <malloc/malloc.h>
#include <os/lock.h>

#include "internal.h"
#include "ring.h"

static int g_pass = 0, g_fail = 0;
static void check(const char *name, int ok, double value, double limit) {
  printf("%s %s %.6g %.6g\n", ok ? "PASS" : "FAIL", name, value, limit);
  if (ok) g_pass++; else g_fail++;
}
static void check_true(const char *name, int ok) { check(name, ok, ok ? 1 : 0, 1); }

/* The rt_guard.dylib API, found at run time. */
static struct {
  int (*add)(pthread_t);
  void (*clear)(void);
  void (*counts)(uint64_t *, uint64_t *, uint64_t *);
  void (*reset)(void);
  uint64_t (*waits)(int slot);
} G;

/* Exported by libsystem_platform (not in the public headers); Accelerate takes this lock. */
void os_unfair_lock_lock_with_options(os_unfair_lock_t lock, uint32_t options);

static uint32_t g_rng = 2463534242u;
static uint32_t xs32(uint32_t *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static float urand(uint32_t *s) { return (float)((xs32(s) >> 8) * (1.0 / 16777216.0)) * 2.0f - 1.0f; }

typedef struct {
  es_engine *e;
  es_stream *direct;          /* a fourth stream, hopped directly on the audio thread */
  const es_cond *c0;
  uint64_t warm_hops, rt_hops;
  /* results */
  uint64_t warm_allocs, warm_frees, warm_locks;
  uint64_t callbacks, direct_hops, measured_hops;
  int timed_out;
  int audio_slot;             /* the audio thread's slot in the guard table (for its wait count) */
  uint64_t wakes_start, wakes;   /* worker wake-ups: counter at the window start, and during the window */
  _Atomic int measuring;      /* audio -> control: the measured window has started */
  _Atomic int stop;           /* audio -> control: the measured window is over */
  /* control thread results */
  es_cond **conds;            /* 3 prebuilt conditioning sets (read-only while the test runs) */
  uint64_t ctl_iters, ctl_accepted, ctl_busy, ctl_nulls, ctl_resets, ctl_params, ctl_probes, ctl_probe_results;
} rt_job;

static void *audio_main(void *arg) {
  rt_job *j = arg;
  es_engine *e = j->e;
  static const int sizes[5] = { 16, 64, 128, 480, 1104 };
  static float in[1104], out[1104], tmp[ES_HOP], heads[2];
  uint32_t rng = 12345u;
  mach_timebase_info_data_t tb;
  mach_timebase_info(&tb);
  uint64_t five_s = (uint64_t)(5e9 * tb.denom / tb.numer);
  es_stats st;
  int phase = 0;                                   /* 0 warm-up, 1 measured */
  uint64_t h_start = 0;

  /* This test thread's own first BLAS call allocates once inside Accelerate (the worker makes its
     own at thread start, rt_thread.c). Make it here, before guarding, with one direct hop. */
  for (int i = 0; i < ES_HOP; i++) in[i] = 0.0f;
  es_stream_hop(j->direct, j->c0, in, tmp, heads);
  G.reset();
  j->audio_slot = G.add(pthread_self());
  G.add(e->thread);
  for (;;) {
    uint64_t hops = atomic_load_explicit(&e->s_hops, memory_order_relaxed);
    if (phase == 0 && hops >= j->warm_hops) {      /* end of warm-up: keep its counts, start fresh */
      j->wakes_start = atomic_load(&e->worker_wakes);   /* read before the reset (see the waits row) */
      G.counts(&j->warm_allocs, &j->warm_frees, &j->warm_locks);
      G.reset();
      phase = 1;
      h_start = hops;
      atomic_store(&j->measuring, 1);              /* the control thread starts its changes now */
    }
    if (phase == 1 && hops - h_start >= j->rt_hops) break;

    /* One callback. */
    int n = sizes[xs32(&rng) % 5];
    for (int i = 0; i < n; i++) in[i] = 0.3f * urand(&rng);
    es_engine_render(e, in, out, n);
    j->callbacks++;
    es_engine_get_stats(e, &st);
    (void)es_engine_input_position(e);
    if (j->callbacks % 997 == 0) es_engine_cue(e, 1 + (int)(j->callbacks % 3));
    if (j->callbacks % 1000 == 500) es_engine_set_mute(e, 1);
    if (j->callbacks % 1000 == 520) es_engine_set_mute(e, 0);
    if (j->callbacks % 8 == 0) {
      es_stream_hop(j->direct, j->c0, in, tmp, heads);
      j->direct_hops++;
    }

    /* Wait for the worker: every complete hop of the input written so far has run. */
    uint64_t want = es_engine_input_position(e) / ES_HOP, t0 = mach_absolute_time();
    while (atomic_load_explicit(&e->s_hops, memory_order_relaxed) < want) {
      sched_yield();
      if (mach_absolute_time() - t0 > five_s) { j->timed_out = 1; break; }
    }
    if (j->timed_out) break;
  }
  G.clear();
  j->wakes = atomic_load(&e->worker_wakes) - j->wakes_start;   /* read after the clear */
  atomic_store(&j->stop, 1);
  j->measured_hops = atomic_load(&e->s_hops) - h_start;
  return NULL;
}

/* The control thread (not guarded): conditioning swaps, params changes and probes while the audio
   thread and the worker run under the guard. */
static void *control_main(void *arg) {
  rt_job *j = arg;
  es_engine *e = j->e;
  es_params p;
  es_params_default(&p);
  while (!atomic_load(&j->measuring) && !atomic_load(&j->stop)) usleep(200);
  for (uint64_t it = 0; !atomic_load(&j->stop); it++) {
    int slot = (int)(it % 3);
    int use_null = it % 7 == 3;                    /* empty the slot now and then */
    int reset = (int)((it / 3) % 2);
    es_status st = es_engine_set_cond(e, slot, use_null ? NULL : j->conds[(it / 3) % 3], reset);
    if (st == ES_ERR_BUSY) {
      j->ctl_busy++;
    } else if (st == ES_OK) {
      j->ctl_accepted++;
      if (use_null) j->ctl_nulls++;
      if (reset) j->ctl_resets++;
    }
    p.bypass = (int)((it / 50) % 2);               /* the seqlock writer, with real changes */
    p.room_db = it % 3 ? -24.0f : -60.0f;
    p.agc = it % 5 != 0;
    p.slot_gain[1] = (float)(it % 10) / 10.0f;
    p.slot_gain[2] = 1.0f - p.slot_gain[1];
    es_engine_set_params(e, &p);
    j->ctl_params++;
    if (it % 400 == 10 && es_engine_probe_start(e) == ES_OK) j->ctl_probes++;
    if (it % 100 == 99) {                          /* GCC-PHAT runs here, off the RT threads */
      es_probe_result r;
      es_status pr = es_engine_probe_result(e, &r);
      if (pr == ES_OK || pr == ES_ERR_PROBE_WEAK) j->ctl_probe_results++;
    }
    j->ctl_iters++;
    usleep(1000);
  }
  return NULL;
}

/* A conditioning set with random values (the RT rules do not depend on them). */
static es_cond *random_cond(int n) {
  es_cond *c = es_cond_new();
  for (int i = 0; i < 4096; i++) c->film[i] = 0.05f * urand(&g_rng);
  for (int h = 0; h < 4; h++)
    for (int t = 0; t < 68; t++)
      for (int d = 0; d < 16; d++) {
        c->k[h][t][d] = urand(&g_rng);
        c->v[h][t][d] = urand(&g_rng);
      }
  c->n_tokens = n;
  return c;
}

/* Counts added since the previous call (allocs, frees, locks). Call it once right after G.reset(). */
static void delta(uint64_t *da, uint64_t *df, uint64_t *dl) {
  static uint64_t pa, pf, pl;
  uint64_t a, f, l;
  G.counts(&a, &f, &l);
  *da = a - pa;
  *df = f - pf;
  *dl = l - pl;
  pa = a;
  pf = f;
  pl = l;
}

/* Prove the guard counts on a guarded thread and ignores others (so a 0 below means something).
   One small step per kind of call, each checked on its own (nothing else, not even printf, runs
   between the steps). A call may make further counted calls inside the system libraries (the first
   malloc_type_malloc makes 2 allocations), so every step checks "at least 1". */
static void guard_self_check(void) {
  pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
  pthread_rwlock_t rw = PTHREAD_RWLOCK_INITIALIZER;
  os_unfair_lock ul = OS_UNFAIR_LOCK_INIT, ul2 = OS_UNFAIR_LOCK_INIT;
  dispatch_semaphore_t ds = dispatch_semaphore_create(1);   /* made before guarding (it allocates) */
  enum { MALLOC, MEMALIGN, TYPED, MUTEX, TRYLOCK, RWLOCK, UNFAIR, UNFAIR_OPT, NSTEP };
  uint64_t da[NSTEP], df[NSTEP], dl[NSTEP], qa, qf, ql;
  G.reset();
  delta(&qa, &qf, &ql);
  void *volatile p = malloc(64);                   /* main is not guarded yet: not counted */
  free(p);                                         /* (volatile: clang may not drop the pair) */
  delta(&qa, &qf, &ql);
  int slot = G.add(pthread_self());

  p = malloc(64);
  free(p);
  delta(&da[MALLOC], &df[MALLOC], &dl[MALLOC]);
  void *qq = NULL;
  if (posix_memalign(&qq, 64, 128) == 0) {
    void *volatile q = qq;
    free(q);
  }
  delta(&da[MEMALIGN], &df[MEMALIGN], &dl[MEMALIGN]);
  void *volatile t = malloc_type_malloc(64, 0);    /* the typed allocator */
  malloc_type_free(t, 0);
  delta(&da[TYPED], &df[TYPED], &dl[TYPED]);
  pthread_mutex_lock(&mu);
  pthread_mutex_unlock(&mu);
  delta(&da[MUTEX], &df[MUTEX], &dl[MUTEX]);
  if (pthread_mutex_trylock(&mu) == 0) pthread_mutex_unlock(&mu);
  delta(&da[TRYLOCK], &df[TRYLOCK], &dl[TRYLOCK]);
  pthread_rwlock_rdlock(&rw);
  pthread_rwlock_unlock(&rw);
  delta(&da[RWLOCK], &df[RWLOCK], &dl[RWLOCK]);
  os_unfair_lock_lock(&ul);
  os_unfair_lock_unlock(&ul);
  delta(&da[UNFAIR], &df[UNFAIR], &dl[UNFAIR]);
  os_unfair_lock_lock_with_options(&ul2, 0);
  os_unfair_lock_unlock(&ul2);
  delta(&da[UNFAIR_OPT], &df[UNFAIR_OPT], &dl[UNFAIR_OPT]);
  dispatch_semaphore_wait(ds, DISPATCH_TIME_NOW);  /* a wait, counted for this thread's slot */
  uint64_t w = G.waits(slot);
  G.clear();
  G.reset();
  dispatch_semaphore_signal(ds);
  dispatch_release(ds);

  int ok = qa == 0 && qf == 0 && ql == 0 && w == 1;
  for (int k = MALLOC; k <= TYPED; k++) ok = ok && da[k] >= 1 && df[k] >= 1;
  for (int k = MUTEX; k < NSTEP; k++) ok = ok && dl[k] >= 1;
  printf("INFO rt_guard self check: unguarded %llu/%llu/%llu; malloc %llu/%llu, posix_memalign %llu/%llu, "
         "malloc_type %llu/%llu (allocs/frees); locks: mutex %llu, trylock %llu, rwlock %llu, unfair %llu, "
         "unfair+options %llu; waits %llu\n", (unsigned long long)qa, (unsigned long long)qf,
         (unsigned long long)ql, (unsigned long long)da[MALLOC], (unsigned long long)df[MALLOC],
         (unsigned long long)da[MEMALIGN], (unsigned long long)df[MEMALIGN], (unsigned long long)da[TYPED],
         (unsigned long long)df[TYPED], (unsigned long long)dl[MUTEX], (unsigned long long)dl[TRYLOCK],
         (unsigned long long)dl[RWLOCK], (unsigned long long)dl[UNFAIR], (unsigned long long)dl[UNFAIR_OPT],
         (unsigned long long)w);
  check_true("rt_guard_counts_guarded_thread_only", ok);
}

int main(int argc, char **argv) {
  printf("test_rt (packages E-2, E-3), hop path %s\n", es_hop_path);
  G.add = (int (*)(pthread_t))dlsym(RTLD_DEFAULT, "rt_guard_add");
  G.clear = (void (*)(void))dlsym(RTLD_DEFAULT, "rt_guard_clear");
  G.counts = (void (*)(uint64_t *, uint64_t *, uint64_t *))dlsym(RTLD_DEFAULT, "rt_guard_counts");
  G.reset = (void (*)(void))dlsym(RTLD_DEFAULT, "rt_guard_reset");
  G.waits = (uint64_t (*)(int))dlsym(RTLD_DEFAULT, "rt_guard_waits");
  if (!G.add || !G.clear || !G.counts || !G.reset || !G.waits) {
    printf("FAIL rt_guard_loaded 0 1 (run with DYLD_INSERT_LIBRARIES=build/rt_guard.dylib)\n");
    return 1;
  }
  guard_self_check();

  const char *env = getenv("RT_HOPS");
  uint64_t rt_hops = env ? strtoull(env, NULL, 10) : 100000;
  es_model *m = NULL;
  const char *model_name = "synthetic";
  if (argc == 3 && strcmp(argv[1], "--export") == 0) {
    char path[1024], err[256];
    snprintf(path, sizeof(path), "%s/earshot.esw", argv[2]);
    if (es_model_load_file(path, &m, err, sizeof(err)) != ES_OK) {
      printf("FAIL model_load 0 1 (%s)\n", err);
      return 1;
    }
    model_name = argv[2];
  } else if (es_model_synthetic(21, 0, &m) != ES_OK) {
    printf("FAIL model_synthetic 0 1\n");
    return 1;
  }

  /* Engine: 48 kHz, 64-frame grant (callbacks still vary up to 1,104), 3 instances, all filled. */
  es_engine_config cfg = { 48000, 64, 4096, 3, 43 };
  es_engine *e = NULL;
  if (es_engine_new(m, &cfg, &e) != ES_OK) {
    printf("FAIL engine_new 0 1\n");
    return 1;
  }
  es_cond *c[3] = { random_cond(20), random_cond(37), random_cond(68) };
  for (int s = 0; s < 3; s++) es_engine_set_cond(e, s, c[s], 1);
  es_params p;
  es_params_default(&p);
  p.slot_gain[1] = p.slot_gain[2] = 0.5f;
  p.room_db = -24;
  es_engine_set_params(e, &p);
  es_engine_set_mute(e, 0);

  rt_job j;
  memset(&j, 0, sizeof(j));
  j.e = e;
  j.direct = es_stream_new(m);
  j.c0 = c[0];
  j.warm_hops = 100;
  j.rt_hops = rt_hops;
  j.conds = c;

  es_status st = es_engine_start_worker(e, NULL);
  check_true("worker_started", st == ES_OK);
  if (st != ES_OK) return 1;
  uint64_t t0 = mach_absolute_time();
  pthread_t audio, control;
  pthread_create(&control, NULL, control_main, &j);
  pthread_create(&audio, NULL, audio_main, &j);
  pthread_join(audio, NULL);
  pthread_join(control, NULL);
  uint64_t t1 = mach_absolute_time();
  es_engine_stop_worker(e);

  uint64_t a, f, l;
  G.counts(&a, &f, &l);
  uint64_t audio_waits = G.waits(j.audio_slot), worker_waits = G.waits(j.audio_slot == 0 ? 1 : 0);
  es_stats s;
  es_engine_get_stats(e, &s);
  mach_timebase_info_data_t tb;
  mach_timebase_info(&tb);
  double secs = (double)(t1 - t0) * tb.numer / tb.denom / 1e9;
  printf("INFO model %s, 3 instances, %llu measured hops, %llu callbacks, %llu direct hops, %.1f s wall, "
         "underruns %llu, nan_resets %llu, compute_us_max %.0f, worker workgroup rc %d\n",
         model_name, (unsigned long long)j.measured_hops, (unsigned long long)j.callbacks,
         (unsigned long long)j.direct_hops, secs, (unsigned long long)s.underruns,
         (unsigned long long)s.nan_resets, s.compute_us_max, e->wg_join_result);
  printf("INFO warm-up (first 100 hops, not gated): %llu allocs, %llu frees, %llu locks\n",
         (unsigned long long)j.warm_allocs, (unsigned long long)j.warm_frees, (unsigned long long)j.warm_locks);
  check("rt_warmup_allocations_and_locks", j.warm_allocs + j.warm_frees + j.warm_locks == 0,
        (double)(j.warm_allocs + j.warm_frees + j.warm_locks), 0);
  check_true("rt_no_timeout", !j.timed_out);
  check("rt_hops_done", j.measured_hops >= rt_hops, (double)j.measured_hops, (double)rt_hops);
  check("rt_allocations", a == 0, (double)a, 0);
  check("rt_frees", f == 0, (double)f, 0);
  check("rt_lock_calls", l == 0, (double)l, 0);
  printf("INFO waits in the measured window: audio thread %llu, worker %llu, worker wake-ups %llu\n",
         (unsigned long long)audio_waits, (unsigned long long)worker_waits, (unsigned long long)j.wakes);
  check("rt_audio_thread_waits", audio_waits == 0, (double)audio_waits, 0);
  /* The worker's only allowed wait is the idle semaphore_wait at the top of its loop, one per wake-up.
     Every wait counted in the window is followed by a wake-up inside it, except at most the last one
     (the wake counter is read before the guard reset and after the guard clear), so waits <= wakes + 1.
     A wait inside es_engine_run_hops would add a second wait per wake-up and fail this row (E-3: closes
     the E-2 review note that worker waits were only reported). */
  check("rt_worker_waits_only_idle_wait_per_wakeup", worker_waits <= j.wakes + 1, (double)worker_waits,
        (double)(j.wakes + 1));
  printf("INFO control thread: %llu iterations, set_cond %llu accepted (%llu NULL, %llu with reset), %llu BUSY, "
         "%llu params changes, %llu probes started, %llu probe results; worker applied %llu resets on slot 0\n",
         (unsigned long long)j.ctl_iters, (unsigned long long)j.ctl_accepted, (unsigned long long)j.ctl_nulls,
         (unsigned long long)j.ctl_resets, (unsigned long long)j.ctl_busy, (unsigned long long)j.ctl_params,
         (unsigned long long)j.ctl_probes, (unsigned long long)j.ctl_probe_results,
         (unsigned long long)e->n_resets[0]);
  /* The window must really have covered the swap, reset, empty-slot, params and probe paths. */
  check_true("rt_control_paths_covered", j.ctl_accepted >= 30 && j.ctl_nulls >= 3 && j.ctl_resets >= 3 &&
             j.ctl_params >= 30 && j.ctl_probes >= 1);

  es_stream_free(j.direct);
  for (int i = 0; i < 3; i++) es_cond_free(c[i]);
  es_engine_free(e);
  es_model_free(m);
  printf("test_rt: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
