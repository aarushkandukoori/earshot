/* rt_guard.c - allocation, lock and wait counter for test_rt (20-engine 14.3, contract 4.1). Builds
 * build/rt_guard.dylib.
 *
 * How it works. The dylib is loaded into test_rt with DYLD_INSERT_LIBRARIES. Its __DATA,__interpose
 * section tells dyld: "whenever any other image calls malloc (or free, or pthread_mutex_lock, ...),
 * call my hook instead". Each hook checks whether the calling thread is one of up to 4 guarded threads;
 * if so it adds 1 to a global atomic counter. Then it calls the real function (calls made from inside
 * this dylib are never redirected, so `malloc` here is the real one).
 *
 * The thread check is a loop over a 4-entry table compared with pthread_self(), which only reads a
 * register: no thread-local storage, so a hook can never allocate by itself.
 *
 * What is counted (E-2 review widened the list: Accelerate takes os_unfair_lock_lock_with_options, and
 * the typed malloc_type_* calls are what newer compilers emit for malloc):
 *   allocs  malloc, calloc, realloc, valloc, posix_memalign, aligned_alloc, malloc_zone_*, and the
 *           malloc_type_* versions of all of these
 *   frees   free, malloc_zone_free, malloc_type_free, malloc_type_zone_free
 *   locks   pthread_mutex_lock / trylock, pthread_rwlock_rdlock / wrlock, os_unfair_lock_lock,
 *           os_unfair_lock_lock_with_options, dispatch_async / dispatch_sync (and their _f forms)
 *   waits   semaphore_wait, semaphore_timedwait, dispatch_semaphore_wait, counted per guarded thread
 *           (the worker thread is allowed exactly one kind of wait: its idle semaphore_wait between
 *           callbacks, outside es_engine_run_hops; the audio thread may never wait)
 *
 * Threads: the hooks run on every thread of the process; test_rt (one thread at a time, outside the
 * measured window) calls rt_guard_add / rt_guard_clear / rt_guard_counts / rt_guard_waits, looked up
 * with dlsym.
 */
#include <dispatch/dispatch.h>
#include <malloc/malloc.h>
#include <mach/mach.h>
#include <os/lock.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

/* Not in the public SDK headers, but exported by libsystem_platform and used inside Accelerate. */
void os_unfair_lock_lock_with_options(os_unfair_lock_t lock, uint32_t options);

/* ---- Counters and the guarded-thread table ---- */
static _Atomic uintptr_t g_threads[4];
static _Atomic uint64_t g_allocs, g_frees, g_locks;
static _Atomic uint64_t g_waits[4];                /* per table slot */

/* Table slot of the calling thread, or -1 if it is not guarded. */
static int guarded_slot(void) {
  uintptr_t me = (uintptr_t)pthread_self();
  for (int i = 0; i < 4; i++)
    if (atomic_load_explicit(&g_threads[i], memory_order_relaxed) == me) return i;
  return -1;
}
static void count(_Atomic uint64_t *c) {
  if (guarded_slot() >= 0) atomic_fetch_add_explicit(c, 1, memory_order_relaxed);
}
static void count_wait(void) {
  int i = guarded_slot();
  if (i >= 0) atomic_fetch_add_explicit(&g_waits[i], 1, memory_order_relaxed);
}

/* ---- API for test_rt (found with dlsym, so test_rt also runs, and fails clearly, without us) ---- */
/* Adds a thread to the table; returns its slot (0..3), or -1 if the table is full. */
__attribute__((visibility("default"))) int rt_guard_add(pthread_t t) {
  for (int i = 0; i < 4; i++) {
    uintptr_t expect = 0;
    if (atomic_compare_exchange_strong(&g_threads[i], &expect, (uintptr_t)t)) return i;
  }
  return -1;
}
__attribute__((visibility("default"))) void rt_guard_clear(void) {
  for (int i = 0; i < 4; i++) atomic_store(&g_threads[i], 0);
}
__attribute__((visibility("default"))) void rt_guard_counts(uint64_t *allocs, uint64_t *frees, uint64_t *locks) {
  *allocs = atomic_load(&g_allocs);
  *frees = atomic_load(&g_frees);
  *locks = atomic_load(&g_locks);
}
/* Waits made by the thread in table slot `slot` since the last reset. */
__attribute__((visibility("default"))) uint64_t rt_guard_waits(int slot) {
  return slot >= 0 && slot < 4 ? atomic_load(&g_waits[slot]) : 0;
}
__attribute__((visibility("default"))) void rt_guard_reset(void) {
  atomic_store(&g_allocs, 0);
  atomic_store(&g_frees, 0);
  atomic_store(&g_locks, 0);
  for (int i = 0; i < 4; i++) atomic_store(&g_waits[i], 0);
}

/* ---- The hooks: count, then call the real function ---- */
/* Allocations */
static void *h_malloc(size_t n) { count(&g_allocs); return malloc(n); }
static void *h_calloc(size_t k, size_t n) { count(&g_allocs); return calloc(k, n); }
static void *h_realloc(void *p, size_t n) { count(&g_allocs); return realloc(p, n); }
static void *h_valloc(size_t n) { count(&g_allocs); return valloc(n); }
static int h_posix_memalign(void **p, size_t a, size_t n) { count(&g_allocs); return posix_memalign(p, a, n); }
static void *h_aligned_alloc(size_t a, size_t n) { count(&g_allocs); return aligned_alloc(a, n); }
static void *h_zone_malloc(malloc_zone_t *z, size_t n) { count(&g_allocs); return malloc_zone_malloc(z, n); }
static void *h_zone_calloc(malloc_zone_t *z, size_t k, size_t n) { count(&g_allocs); return malloc_zone_calloc(z, k, n); }
static void *h_zone_realloc(malloc_zone_t *z, void *p, size_t n) { count(&g_allocs); return malloc_zone_realloc(z, p, n); }
static void *h_zone_memalign(malloc_zone_t *z, size_t a, size_t n) { count(&g_allocs); return malloc_zone_memalign(z, a, n); }
/* Typed allocations (malloc_type_id_t says what kind of object it is; we only count) */
static void *h_t_malloc(size_t n, malloc_type_id_t t) { count(&g_allocs); return malloc_type_malloc(n, t); }
static void *h_t_calloc(size_t k, size_t n, malloc_type_id_t t) { count(&g_allocs); return malloc_type_calloc(k, n, t); }
static void *h_t_realloc(void *p, size_t n, malloc_type_id_t t) { count(&g_allocs); return malloc_type_realloc(p, n, t); }
static void *h_t_valloc(size_t n, malloc_type_id_t t) { count(&g_allocs); return malloc_type_valloc(n, t); }
static void *h_t_aligned_alloc(size_t a, size_t n, malloc_type_id_t t) { count(&g_allocs); return malloc_type_aligned_alloc(a, n, t); }
static int h_t_posix_memalign(void **p, size_t a, size_t n, malloc_type_id_t t) {
  count(&g_allocs);
  return malloc_type_posix_memalign(p, a, n, t);
}
static void *h_t_zone_malloc(malloc_zone_t *z, size_t n, malloc_type_id_t t) {
  count(&g_allocs);
  return malloc_type_zone_malloc(z, n, t);
}
static void *h_t_zone_calloc(malloc_zone_t *z, size_t k, size_t n, malloc_type_id_t t) {
  count(&g_allocs);
  return malloc_type_zone_calloc(z, k, n, t);
}
static void *h_t_zone_realloc(malloc_zone_t *z, void *p, size_t n, malloc_type_id_t t) {
  count(&g_allocs);
  return malloc_type_zone_realloc(z, p, n, t);
}
static void *h_t_zone_memalign(malloc_zone_t *z, size_t a, size_t n, malloc_type_id_t t) {
  count(&g_allocs);
  return malloc_type_zone_memalign(z, a, n, t);
}
/* Frees */
static void h_free(void *p) { count(&g_frees); free(p); }
static void h_zone_free(malloc_zone_t *z, void *p) { count(&g_frees); malloc_zone_free(z, p); }
static void h_t_free(void *p, malloc_type_id_t t) { count(&g_frees); malloc_type_free(p, t); }
static void h_t_zone_free(malloc_zone_t *z, void *p, malloc_type_id_t t) { count(&g_frees); malloc_type_zone_free(z, p, t); }
/* Locks, and dispatch calls (they can block or allocate) */
static int h_mutex_lock(pthread_mutex_t *m) { count(&g_locks); return pthread_mutex_lock(m); }
static int h_mutex_trylock(pthread_mutex_t *m) { count(&g_locks); return pthread_mutex_trylock(m); }
static int h_rw_rdlock(pthread_rwlock_t *l) { count(&g_locks); return pthread_rwlock_rdlock(l); }
static int h_rw_wrlock(pthread_rwlock_t *l) { count(&g_locks); return pthread_rwlock_wrlock(l); }
static void h_unfair_lock(os_unfair_lock_t l) { count(&g_locks); os_unfair_lock_lock(l); }
static void h_unfair_lock_opt(os_unfair_lock_t l, uint32_t o) { count(&g_locks); os_unfair_lock_lock_with_options(l, o); }
static void h_dispatch_async(dispatch_queue_t q, dispatch_block_t b) { count(&g_locks); dispatch_async(q, b); }
static void h_dispatch_sync(dispatch_queue_t q, dispatch_block_t b) { count(&g_locks); dispatch_sync(q, b); }
static void h_dispatch_async_f(dispatch_queue_t q, void *c, dispatch_function_t f) { count(&g_locks); dispatch_async_f(q, c, f); }
static void h_dispatch_sync_f(dispatch_queue_t q, void *c, dispatch_function_t f) { count(&g_locks); dispatch_sync_f(q, c, f); }
/* Waits (per thread) */
static kern_return_t h_sem_wait(semaphore_t s) { count_wait(); return semaphore_wait(s); }
static kern_return_t h_sem_timedwait(semaphore_t s, mach_timespec_t t) { count_wait(); return semaphore_timedwait(s, t); }
static intptr_t h_dsem_wait(dispatch_semaphore_t s, dispatch_time_t t) { count_wait(); return dispatch_semaphore_wait(s, t); }

/* One __interpose entry: {replacement, original}. */
typedef struct { const void *replacement, *original; } interpose_t;
#define HOOK(h, f) { (const void *)(h), (const void *)(f) }
__attribute__((used, section("__DATA,__interpose"))) static const interpose_t g_interpose[] = {
  HOOK(h_malloc, malloc), HOOK(h_calloc, calloc), HOOK(h_realloc, realloc), HOOK(h_valloc, valloc),
  HOOK(h_posix_memalign, posix_memalign), HOOK(h_aligned_alloc, aligned_alloc),
  HOOK(h_zone_malloc, malloc_zone_malloc), HOOK(h_zone_calloc, malloc_zone_calloc),
  HOOK(h_zone_realloc, malloc_zone_realloc), HOOK(h_zone_memalign, malloc_zone_memalign),
  HOOK(h_t_malloc, malloc_type_malloc), HOOK(h_t_calloc, malloc_type_calloc),
  HOOK(h_t_realloc, malloc_type_realloc), HOOK(h_t_valloc, malloc_type_valloc),
  HOOK(h_t_aligned_alloc, malloc_type_aligned_alloc), HOOK(h_t_posix_memalign, malloc_type_posix_memalign),
  HOOK(h_t_zone_malloc, malloc_type_zone_malloc), HOOK(h_t_zone_calloc, malloc_type_zone_calloc),
  HOOK(h_t_zone_realloc, malloc_type_zone_realloc), HOOK(h_t_zone_memalign, malloc_type_zone_memalign),
  HOOK(h_free, free), HOOK(h_zone_free, malloc_zone_free), HOOK(h_t_free, malloc_type_free),
  HOOK(h_t_zone_free, malloc_type_zone_free),
  HOOK(h_mutex_lock, pthread_mutex_lock), HOOK(h_mutex_trylock, pthread_mutex_trylock),
  HOOK(h_rw_rdlock, pthread_rwlock_rdlock), HOOK(h_rw_wrlock, pthread_rwlock_wrlock),
  HOOK(h_unfair_lock, os_unfair_lock_lock), HOOK(h_unfair_lock_opt, os_unfair_lock_lock_with_options),
  HOOK(h_dispatch_async, dispatch_async), HOOK(h_dispatch_sync, dispatch_sync),
  HOOK(h_dispatch_async_f, dispatch_async_f), HOOK(h_dispatch_sync_f, dispatch_sync_f),
  HOOK(h_sem_wait, semaphore_wait), HOOK(h_sem_timedwait, semaphore_timedwait),
  HOOK(h_dsem_wait, dispatch_semaphore_wait),
};
