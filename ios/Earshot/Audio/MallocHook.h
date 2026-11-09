/*
 * MallocHook.h - a debug-build-only allocation counter for the two real-time threads
 * (contract 11 "Lead integration", 30-ios 15.3 "Compute"). Owned by WP-IOS.
 *
 * What it proves: the engine's worker thread and the RemoteIO render thread never call
 * malloc or free while audio runs (contract 4.1). DYLD interposition is impossible on the
 * phone, so the app installs libmalloc's `malloc_logger` hook instead. libmalloc calls that
 * hook after every malloc, calloc, realloc and free in the whole process; the hook asks
 * "is this one of the armed threads?" and, if so, adds one to that thread's counter.
 *
 *   slot 0 (MH_SLOT_RENDER) - the RemoteIO render thread (its id comes from eh_render_thread_id)
 *   slot 1 (MH_SLOT_WORKER) - the engine worker, found by its name "earshot.worker"
 *   slot 2 (MH_SLOT_CHECK)  - used only by mh_self_test and the unit tests
 *
 * Threads: mh_install, mh_uninstall, mh_arm, mh_self_test and mh_find_thread_named are for
 * the main thread (Phase 0 runner, Diagnostics). The hook itself runs on every thread and
 * only reads its own thread id and does relaxed atomic adds, so it never blocks or allocates.
 *
 * Release builds: the whole hook is compiled out (EARSHOT_MALLOC_HOOK is defined only in the
 * Debug configuration by project.yml). The functions still exist and do nothing:
 * mh_compiled_in() returns 0 and mh_install() returns 0.
 */
#ifndef EARSHOT_MALLOCHOOK_H
#define EARSHOT_MALLOCHOOK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MH_SLOTS 3
#define MH_SLOT_RENDER 0
#define MH_SLOT_WORKER 1
#define MH_SLOT_CHECK 2

/* 1 when this build contains the hook (Debug), 0 when it was compiled out (Release). */
int mh_compiled_in(void);

/* Installs the hook (keeps and calls any logger that was already there). 1 = installed. */
int mh_install(void);
/* Puts the previous logger back. Counters keep their values. */
void mh_uninstall(void);
int mh_installed(void);

/* Arms `slot` for the thread with this id (pthread_threadid_np / THREAD_IDENTIFIER_INFO);
 * thread_id 0 disarms the slot. Counters are not reset by arming. */
void mh_arm(int slot, uint64_t thread_id);
uint64_t mh_armed(int slot);

/* Allocation events (malloc, calloc, realloc) and free events counted on `slot` so far, and
 * the size of the newest counted allocation (a hint for whoever has to find it). */
void mh_counts(int slot, uint64_t *allocs, uint64_t *frees, uint64_t *last_size);
/* Sets every counter back to 0. */
void mh_reset(void);

/* The calling thread's id (the same number THREAD_IDENTIFIER_INFO reports). */
uint64_t mh_current_thread_id(void);

/* Proves the hook really sees allocations on this device: arms MH_SLOT_CHECK for the calling
 * thread, makes one malloc and one free, disarms, and returns 1 when both were counted.
 * Needs mh_install() first. A 0 means every "0 allocations" result would be meaningless. */
int mh_self_test(void);

/* Looks through the app's threads for one named `name` (for example "earshot.worker").
 * Returns 1 and fills *thread_id and *time_constraint (1 when the thread runs with the
 * real-time THREAD_TIME_CONSTRAINT_POLICY) when found, else 0. Main thread only. */
int mh_find_thread_named(const char *name, uint64_t *thread_id, int *time_constraint);

#ifdef __cplusplus
}
#endif

#endif /* EARSHOT_MALLOCHOOK_H */
