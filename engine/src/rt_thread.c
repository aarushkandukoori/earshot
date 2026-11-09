/* rt_thread.c - the real-time worker thread: start, setup, loop and stop (20-engine 8.4, contract 4.1
 * and 4.2 "Worker").
 *
 * Threads: es_engine_start_worker and es_engine_stop_worker are non-RT (the app's control thread).
 * The worker thread they create runs worker_main: a one-time setup, then a loop that sleeps on the
 * engine's semaphore and runs es_engine_run_hops (pipeline.c) each time the render signals it.
 *
 * Setup, in this order (20-engine 8.4):
 *   1. name the thread "earshot.worker";
 *   2. THREAD_TIME_CONSTRAINT_POLICY with the period / computation / constraint es_engine_new computed
 *      from the config (failure: report ES_ERR_THREAD and exit);
 *   3. join the RemoteIO workgroup if one was given (a failure is recorded, the worker runs unjoined);
 *   4. flush denormals to zero (FPCR.FZ, bit 24) so tiny numbers never slow the math down;
 *   5. single-threaded BLAS (BLASSetThreading) plus tiny BLAS calls (warm_blas, one per BLAS form the
 *      hop uses): the first call of each form a thread makes allocates once inside Accelerate
 *      (measured on the M4, DECISIONS L), so it must happen here and not inside the first hop;
 *   6. worker_running = 1, then tell the starter the result.
 */
#include <pthread.h>
#include <string.h>
#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <os/workgroup.h>

#include "internal.h"

/* Rule 3.6: set FPCR.FZ on this thread (arm64 only; other CPUs keep their default). */
static void flush_denormals(void) {
#if defined(__aarch64__)
  uint64_t r;
  __asm__ volatile("mrs %0, fpcr" : "=r"(r));
  r |= 1ull << 24;
  __asm__ volatile("msr fpcr, %0" ::"r"(r));
#endif
}

/* Tiny BLAS calls so Accelerate's one-time per-thread allocations happen now, not in the first hop.
   Both calls are REQUIRED, one per BLAS form the hop can use:
     sgemv                  the single-row products, and every 16-row product on the default path
     transposed sgemm       the 16-row products on the opt-in -DES_HOP_SGEMM_XT path (es_linear_xt).
                            Its first call on a fresh thread allocates 2 x 8 bytes of per-thread state,
                            and a tiny sgemv does not take that allocation for it (E-3 review, macOS).
   If a new BLAS form enters the hop, add a tiny call of that form here. */
static void warm_blas(void) {
  _Alignas(64) float w[4 * 4], x[4], y[4], xt[4 * 2], y2[2 * 4];
  for (int i = 0; i < 16; i++) w[i] = 0.0f;
  for (int i = 0; i < 4; i++) x[i] = y[i] = 0.0f;
  for (int i = 0; i < 8; i++) xt[i] = y2[i] = 0.0f;
  cblas_sgemv(CblasRowMajor, CblasTrans, 4, 4, 1.0f, w, 4, x, 1, 1.0f, y, 1);
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 2, 4, 4, 1.0f, xt, 2, w, 4, 1.0f, y2, 4);
}

static void *worker_main(void *arg) {
  es_engine *e = arg;

  /* 1. Name. */
  pthread_setname_np("earshot.worker");

  /* 2. Time-constraint policy (values in Mach ticks, computed by es_engine_new). */
  thread_time_constraint_policy_data_t pol;
  pol.period = (uint32_t)e->pol_period;
  pol.computation = (uint32_t)e->pol_computation;
  pol.constraint = (uint32_t)e->pol_constraint;
  pol.preemptible = 1;
  kern_return_t kr = thread_policy_set(pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
                                       (thread_policy_t)&pol, THREAD_TIME_CONSTRAINT_POLICY_COUNT);
  if (kr != KERN_SUCCESS) {
    e->start_status = ES_ERR_THREAD;
    semaphore_signal(e->start_sem);
    return NULL;
  }

  /* 3. Workgroup (os_workgroup_t from RemoteIO's kAudioOutputUnitProperty_OSWorkgroup). */
  os_workgroup_join_token_s token;
  memset(&token, 0, sizeof(token));
  int joined = 0;
  e->wg_join_result = 0;
  if (e->workgroup) {
    int rc = os_workgroup_join((os_workgroup_t)e->workgroup, &token);
    if (rc == 0) joined = 1;
    else e->wg_join_result = rc;                     /* debug flag; keep going unjoined */
  }

  /* 4. Denormals, 5. BLAS. */
  flush_denormals();
  BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
  warm_blas();

  /* 6. Running. */
  atomic_store_explicit(&e->worker_running, 1, memory_order_release);
  e->start_status = ES_OK;
  semaphore_signal(e->start_sem);

  /* Loop: sleep until the render (or stop) signals, then run every complete hop. */
  for (;;) {
    semaphore_wait(e->sem);
    atomic_fetch_add_explicit(&e->worker_wakes, 1, memory_order_relaxed);   /* lock-free counter */
    if (atomic_load_explicit(&e->worker_stop, memory_order_acquire)) break;
    es_engine_run_hops(e);
  }
  if (joined) os_workgroup_leave((os_workgroup_t)e->workgroup, &token);
  return NULL;
}

es_status es_engine_start_worker(es_engine *e, void *os_workgroup) {
  if (!e) return ES_ERR_ARG;
  if (atomic_load_explicit(&e->worker_running, memory_order_acquire)) return ES_ERR_STATE;
  if (semaphore_create(mach_task_self(), &e->start_sem, SYNC_POLICY_FIFO, 0) != KERN_SUCCESS) return ES_ERR_THREAD;
  atomic_store_explicit(&e->worker_stop, 0, memory_order_relaxed);
  e->workgroup = os_workgroup;
  e->start_status = ES_ERR_THREAD;
  if (pthread_create(&e->thread, NULL, worker_main, e) != 0) {
    semaphore_destroy(mach_task_self(), e->start_sem);
    return ES_ERR_THREAD;
  }
  semaphore_wait(e->start_sem);                      /* the thread reports its setup result */
  semaphore_destroy(mach_task_self(), e->start_sem);
  if (e->start_status != ES_OK) {
    pthread_join(e->thread, NULL);
    return e->start_status;
  }
  return ES_OK;
}

void es_engine_stop_worker(es_engine *e) {
  if (!e || !atomic_load_explicit(&e->worker_running, memory_order_acquire)) return;
  atomic_store_explicit(&e->worker_stop, 1, memory_order_release);
  semaphore_signal(e->sem);
  pthread_join(e->thread, NULL);
  atomic_store_explicit(&e->worker_running, 0, memory_order_release);
  /* Nobody picks up conditioning changes now, so they count as picked up (contract 4.3), unless a
     test drives es_engine_run_hops itself. */
  if (!atomic_load_explicit(&e->manual_worker, memory_order_acquire))
    for (int s = 0; s < e->n_inst; s++)
      atomic_store_explicit(&e->ack[s], atomic_load_explicit(&e->active[s], memory_order_acquire),
                            memory_order_release);
}
