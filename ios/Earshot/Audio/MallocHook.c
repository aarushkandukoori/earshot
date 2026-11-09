/*
 * MallocHook.c - the debug-only malloc counter (see MallocHook.h for the why and the rules).
 * Owned by WP-IOS. Implements contract 11 "Lead integration" (counting malloc hook armed for
 * the worker and render thread ids) and the debug line of 30-ios 15.3 "Compute".
 *
 * How libmalloc reports allocations: when the global pointer `malloc_logger` is not NULL,
 * libmalloc calls it after each operation with a type bit mask:
 *   malloc / calloc  -> ALLOCATE (arg2 = size)
 *   free             -> DEALLOCATE
 *   realloc          -> ALLOCATE | DEALLOCATE (arg3 = new size), so it counts as one of each
 * Instruments' allocation tracking uses the same hook. The symbol is exported by libSystem
 * on iOS (checked in the iOS 26.0 SDK libSystem.B.tbd) but not declared in a public header,
 * so it is declared here.
 *
 * Threads: the hook runs on whatever thread called malloc. It reads that thread's id
 * (pthread_threadid_np on the calling thread is a plain field read, no lock, no syscall),
 * compares it with the 3 armed ids and does relaxed atomic adds. Nothing else.
 */
#include "MallocHook.h"

#ifdef EARSHOT_MALLOC_HOOK

#include <mach/mach.h>
#include <mach/thread_info.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

/* libmalloc's hook type and global (private header stack_logging.h; exported symbol). */
typedef void(mh_logger_t)(uint32_t type, uintptr_t arg1, uintptr_t arg2, uintptr_t arg3, uintptr_t result,
                          uint32_t num_hot_frames_to_skip);
extern mh_logger_t *malloc_logger;

#define MH_TYPE_ALLOCATE 2u   /* stack_logging_type_alloc */
#define MH_TYPE_DEALLOCATE 4u /* stack_logging_type_dealloc */

static _Atomic uint64_t g_armed[MH_SLOTS];  /* thread id per slot, 0 = disarmed */
static _Atomic uint64_t g_allocs[MH_SLOTS];
static _Atomic uint64_t g_frees[MH_SLOTS];
static _Atomic uint64_t g_last_size[MH_SLOTS];
static mh_logger_t *g_previous = NULL; /* logger that was installed before ours (Instruments) */
static int g_installed = 0;            /* main thread only */

/* The hook. Called by libmalloc on every thread, so it must stay tiny and lock free. */
static void mh_logger(uint32_t type, uintptr_t arg1, uintptr_t arg2, uintptr_t arg3, uintptr_t result,
                      uint32_t skip) {
  mh_logger_t *prev = g_previous;
  if (prev != NULL) prev(type, arg1, arg2, arg3, result, skip + 1); /* keep Instruments working */
  if ((type & (MH_TYPE_ALLOCATE | MH_TYPE_DEALLOCATE)) == 0) return;
  uint64_t tid = 0;
  pthread_threadid_np(NULL, &tid);
  if (tid == 0) return;
  for (int i = 0; i < MH_SLOTS; i++) {
    if (atomic_load_explicit(&g_armed[i], memory_order_relaxed) != tid) continue;
    if (type & MH_TYPE_ALLOCATE) {
      atomic_fetch_add_explicit(&g_allocs[i], 1, memory_order_relaxed);
      /* realloc passes the new size in arg3, malloc and calloc pass the size in arg2 */
      uint64_t size = (type & MH_TYPE_DEALLOCATE) ? (uint64_t)arg3 : (uint64_t)arg2;
      atomic_store_explicit(&g_last_size[i], size, memory_order_relaxed);
    }
    if (type & MH_TYPE_DEALLOCATE) atomic_fetch_add_explicit(&g_frees[i], 1, memory_order_relaxed);
  }
}

int mh_compiled_in(void) { return 1; }

int mh_install(void) {
  if (g_installed) return 1;
  mh_logger_t *current = malloc_logger;
  g_previous = (current == mh_logger) ? NULL : current;
  malloc_logger = mh_logger;
  g_installed = 1;
  return 1;
}

void mh_uninstall(void) {
  if (!g_installed) return;
  malloc_logger = g_previous; /* a thread inside mh_logger right now still finishes safely: the
                                 code stays in the app and only reads atomics */
  g_previous = NULL;
  g_installed = 0;
}

int mh_installed(void) { return g_installed; }

void mh_arm(int slot, uint64_t thread_id) {
  if (slot < 0 || slot >= MH_SLOTS) return;
  atomic_store_explicit(&g_armed[slot], thread_id, memory_order_relaxed);
}

uint64_t mh_armed(int slot) {
  if (slot < 0 || slot >= MH_SLOTS) return 0;
  return atomic_load_explicit(&g_armed[slot], memory_order_relaxed);
}

void mh_counts(int slot, uint64_t *allocs, uint64_t *frees, uint64_t *last_size) {
  int ok = slot >= 0 && slot < MH_SLOTS;
  if (allocs) *allocs = ok ? atomic_load_explicit(&g_allocs[slot], memory_order_relaxed) : 0;
  if (frees) *frees = ok ? atomic_load_explicit(&g_frees[slot], memory_order_relaxed) : 0;
  if (last_size) *last_size = ok ? atomic_load_explicit(&g_last_size[slot], memory_order_relaxed) : 0;
}

void mh_reset(void) {
  for (int i = 0; i < MH_SLOTS; i++) {
    atomic_store_explicit(&g_allocs[i], 0, memory_order_relaxed);
    atomic_store_explicit(&g_frees[i], 0, memory_order_relaxed);
    atomic_store_explicit(&g_last_size[i], 0, memory_order_relaxed);
  }
}

uint64_t mh_current_thread_id(void) {
  uint64_t tid = 0;
  pthread_threadid_np(NULL, &tid);
  return tid;
}

int mh_self_test(void) {
  if (!g_installed) return 0;
  uint64_t a0 = 0, f0 = 0, a1 = 0, f1 = 0;
  uint64_t saved = mh_armed(MH_SLOT_CHECK);
  mh_counts(MH_SLOT_CHECK, &a0, &f0, NULL);
  mh_arm(MH_SLOT_CHECK, mh_current_thread_id());
  /* volatile so the compiler cannot remove the malloc / free pair */
  void *volatile p = malloc(97);
  if (p != NULL) ((volatile char *)p)[0] = 1;
  free(p);
  mh_arm(MH_SLOT_CHECK, saved);
  mh_counts(MH_SLOT_CHECK, &a1, &f1, NULL);
  return (p != NULL && a1 > a0 && f1 > f0) ? 1 : 0;
}

int mh_find_thread_named(const char *name, uint64_t *thread_id, int *time_constraint) {
  if (name == NULL) return 0;
  thread_act_array_t list = NULL;
  mach_msg_type_number_t count = 0;
  if (task_threads(mach_task_self(), &list, &count) != KERN_SUCCESS) return 0;
  int found = 0;
  for (mach_msg_type_number_t i = 0; i < count; i++) {
    thread_extended_info_data_t ext;
    mach_msg_type_number_t n = THREAD_EXTENDED_INFO_COUNT;
    if (!found && thread_info(list[i], THREAD_EXTENDED_INFO, (thread_info_t)&ext, &n) == KERN_SUCCESS &&
        strncmp(ext.pth_name, name, sizeof(ext.pth_name)) == 0) {
      thread_identifier_info_data_t ident;
      n = THREAD_IDENTIFIER_INFO_COUNT;
      if (thread_info(list[i], THREAD_IDENTIFIER_INFO, (thread_info_t)&ident, &n) == KERN_SUCCESS) {
        /* get_default comes back TRUE when the thread has no time-constraint policy of its own */
        thread_time_constraint_policy_data_t pol;
        mach_msg_type_number_t pn = THREAD_TIME_CONSTRAINT_POLICY_COUNT;
        boolean_t get_default = FALSE;
        int rt = thread_policy_get(list[i], THREAD_TIME_CONSTRAINT_POLICY, (thread_policy_t)&pol, &pn,
                                   &get_default) == KERN_SUCCESS && !get_default;
        if (thread_id) *thread_id = ident.thread_id;
        if (time_constraint) *time_constraint = rt;
        found = 1;
      }
    }
    mach_port_deallocate(mach_task_self(), list[i]);
  }
  vm_deallocate(mach_task_self(), (vm_address_t)list, sizeof(thread_act_t) * count);
  return found;
}

#else /* Release: compiled out, every call is a no-op */

int mh_compiled_in(void) { return 0; }
int mh_install(void) { return 0; }
void mh_uninstall(void) {}
int mh_installed(void) { return 0; }
void mh_arm(int slot, uint64_t thread_id) {
  (void)slot;
  (void)thread_id;
}
uint64_t mh_armed(int slot) {
  (void)slot;
  return 0;
}
void mh_counts(int slot, uint64_t *allocs, uint64_t *frees, uint64_t *last_size) {
  (void)slot;
  if (allocs) *allocs = 0;
  if (frees) *frees = 0;
  if (last_size) *last_size = 0;
}
void mh_reset(void) {}
uint64_t mh_current_thread_id(void) { return 0; }
int mh_self_test(void) { return 0; }
int mh_find_thread_named(const char *name, uint64_t *thread_id, int *time_constraint) {
  (void)name;
  (void)thread_id;
  (void)time_constraint;
  return 0;
}

#endif
