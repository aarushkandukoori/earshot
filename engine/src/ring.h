/* ring.h - lock-free rings shared between the audio thread, the worker and readers (20-engine 8.1).
 *
 * Header-only, C11 atomics, no allocation: the caller (es_engine_new) allocates every buffer.
 * Every function here is RT-safe (a few loads, stores and at most two memcpy calls).
 *
 * Three kinds of ring:
 *   es_ring   - single producer, single consumer, float items (in16, out16)
 *   es_ring16 - the same with uint16_t items (per-hop compute timings)
 *   es_hist   - one writer, any number of readers that each keep their own cursor and never remove
 *               anything (the 16 kHz input history and the per-slot output history)
 *
 * How the SPSC ring works: w counts every item ever written and r every item ever read (64-bit, so they
 * never wrap in practice). The item with absolute index i lives at buf[i & mask], so the size must be a
 * power of two. Only the producer stores w and only the consumer stores r. The producer writes the data
 * first and then publishes w with a release store; the consumer loads w with acquire, so once it sees the
 * new w it also sees the data. The same pairing works the other way for r, which tells the producer that
 * a slot is free again.
 */
#ifndef EARSHOT_RING_H
#define EARSHOT_RING_H

#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

/* ---------------------------------------------------------------------------------------------- */
/* Float SPSC ring                                                                                 */
/* ---------------------------------------------------------------------------------------------- */
typedef struct {
  _Alignas(64) _Atomic uint64_t w;   /* total items written; stored only by the producer (release) */
  _Alignas(64) _Atomic uint64_t r;   /* total items read; stored only by the consumer (release) */
  _Alignas(64) float *buf;
  uint32_t size, mask;               /* size is a power of two, mask = size - 1 */
} es_ring;

/* Set up a ring over a caller-owned buffer of `size` floats (a power of two). Not RT (call before use). */
static inline void ring_init(es_ring *q, float *buf, uint32_t size) {
  atomic_store_explicit(&q->w, 0, memory_order_relaxed);
  atomic_store_explicit(&q->r, 0, memory_order_relaxed);
  q->buf = buf;
  q->size = size;
  q->mask = size - 1;
}

/* Consumer side: how many items can be read now. */
static inline uint32_t ring_count(es_ring *q) {
  uint64_t w = atomic_load_explicit(&q->w, memory_order_acquire);
  uint64_t r = atomic_load_explicit(&q->r, memory_order_relaxed);
  return (uint32_t)(w - r);
}

/* Producer side: how many items can be written now. */
static inline uint32_t ring_space(es_ring *q) {
  uint64_t w = atomic_load_explicit(&q->w, memory_order_relaxed);
  uint64_t r = atomic_load_explicit(&q->r, memory_order_acquire);
  return q->size - (uint32_t)(w - r);
}

/* Producer: write n items (the caller checked ring_space). At most two memcpy calls (wrap-around). */
static inline void ring_write(es_ring *q, const float *x, uint32_t n) {
  uint64_t w = atomic_load_explicit(&q->w, memory_order_relaxed);
  uint32_t at = (uint32_t)(w & q->mask);
  uint32_t first = q->size - at;               /* room before the end of the buffer */
  if (first > n) first = n;
  memcpy(q->buf + at, x, sizeof(float) * first);
  memcpy(q->buf, x + first, sizeof(float) * (n - first));
  atomic_store_explicit(&q->w, w + n, memory_order_release);
}

/* Consumer: read n items (the caller checked ring_count). */
static inline void ring_read(es_ring *q, float *x, uint32_t n) {
  uint64_t r = atomic_load_explicit(&q->r, memory_order_relaxed);
  uint32_t at = (uint32_t)(r & q->mask);
  uint32_t first = q->size - at;
  if (first > n) first = n;
  memcpy(x, q->buf + at, sizeof(float) * first);
  memcpy(x + first, q->buf, sizeof(float) * (n - first));
  atomic_store_explicit(&q->r, r + n, memory_order_release);
}

/* Consumer: drop n items without copying them (the caller checked ring_count). */
static inline void ring_skip(es_ring *q, uint32_t n) {
  uint64_t r = atomic_load_explicit(&q->r, memory_order_relaxed);
  atomic_store_explicit(&q->r, r + n, memory_order_release);
}

/* Consumer: total items read so far (the "R" of 20-engine 8.3 step 3). */
static inline uint64_t ring_total_read(es_ring *q) {
  return atomic_load_explicit(&q->r, memory_order_relaxed);
}

/* ---------------------------------------------------------------------------------------------- */
/* uint16_t SPSC ring (per-hop compute timings in microseconds). Same rules as es_ring.            */
/* ---------------------------------------------------------------------------------------------- */
typedef struct {
  _Alignas(64) _Atomic uint64_t w;
  _Alignas(64) _Atomic uint64_t r;
  _Alignas(64) uint16_t *buf;
  uint32_t size, mask;
} es_ring16;

static inline void ring16_init(es_ring16 *q, uint16_t *buf, uint32_t size) {
  atomic_store_explicit(&q->w, 0, memory_order_relaxed);
  atomic_store_explicit(&q->r, 0, memory_order_relaxed);
  q->buf = buf;
  q->size = size;
  q->mask = size - 1;
}

static inline uint32_t ring16_count(es_ring16 *q) {
  uint64_t w = atomic_load_explicit(&q->w, memory_order_acquire);
  uint64_t r = atomic_load_explicit(&q->r, memory_order_relaxed);
  return (uint32_t)(w - r);
}

static inline uint32_t ring16_space(es_ring16 *q) {
  uint64_t w = atomic_load_explicit(&q->w, memory_order_relaxed);
  uint64_t r = atomic_load_explicit(&q->r, memory_order_acquire);
  return q->size - (uint32_t)(w - r);
}

/* Producer: write one item (the caller checked ring16_space >= 1). */
static inline void ring16_write1(es_ring16 *q, uint16_t v) {
  uint64_t w = atomic_load_explicit(&q->w, memory_order_relaxed);
  q->buf[w & q->mask] = v;
  atomic_store_explicit(&q->w, w + 1, memory_order_release);
}

/* Consumer: read up to max items, returns how many were read. */
static inline uint32_t ring16_read(es_ring16 *q, uint16_t *x, uint32_t max) {
  uint32_t n = ring16_count(q);
  if (n > max) n = max;
  uint64_t r = atomic_load_explicit(&q->r, memory_order_relaxed);
  for (uint32_t i = 0; i < n; i++) x[i] = q->buf[(r + i) & q->mask];
  atomic_store_explicit(&q->r, r + n, memory_order_release);
  return n;
}

/* ---------------------------------------------------------------------------------------------- */
/* History ring: one writer, readers with their own cursor (20-engine 8.1).                        */
/*                                                                                                 */
/* The writer overwrites the oldest audio. A reader may only start at most keep =                  */
/* (size - ES_HIST_GUARD) items behind the newest one, so a writer chunk of up to ES_HIST_GUARD     */
/* items that is still being copied (not yet published in w) can never touch what the reader       */
/* copies. After copying, the reader loads w again (w_after). The chunk that may still be in       */
/* flight then covers absolute items [w_after, w_after + ES_HIST_GUARD), which share buffer slots  */
/* with [w_after - size, w_after - keep). So the copy is safe only if it started at or after       */
/* w_after - keep, that is w_after - start <= keep. Otherwise the copy may hold overwritten audio: */
/* it is thrown away (returns 0) and the cursor jumps to the oldest safe position.                 */
/* (Real writer chunks are at most 1,600 items, so the guard also leaves room for several chunks.) */
/* ---------------------------------------------------------------------------------------------- */
#define ES_HIST_GUARD 16384u

typedef struct {
  _Alignas(64) _Atomic uint64_t w;   /* total items written; stored only by the writer (release) */
  _Alignas(64) float *buf;
  uint32_t size, mask;               /* power of two, larger than ES_HIST_GUARD */
} es_hist;

static inline void hist_init(es_hist *h, float *buf, uint32_t size) {
  atomic_store_explicit(&h->w, 0, memory_order_relaxed);
  h->buf = buf;
  h->size = size;
  h->mask = size - 1;
}

/* Writer: append n items (n <= ES_HIST_GUARD). Two memcpy at most, then publish. RT-safe. */
static inline void hist_write(es_hist *h, const float *x, uint32_t n) {
  uint64_t w = atomic_load_explicit(&h->w, memory_order_relaxed);
  uint32_t at = (uint32_t)(w & h->mask);
  uint32_t first = h->size - at;
  if (first > n) first = n;
  memcpy(h->buf + at, x, sizeof(float) * first);
  memcpy(h->buf, x + first, sizeof(float) * (n - first));
  atomic_store_explicit(&h->w, w + n, memory_order_release);
}

/* Anyone: total items written so far (one acquire load, RT-safe). */
static inline uint64_t hist_position(es_hist *h) {
  return atomic_load_explicit(&h->w, memory_order_acquire);
}

/* Reader: copy up to max items starting at *cursor (or at the oldest held item if *cursor is older).
   Returns the number copied and moves *cursor past them. On an overrun returns 0 and sets
   *cursor = w - (size - ES_HIST_GUARD). Skipped items = new cursor - old cursor - returned.
   A cursor ahead of the writer (from the future) returns 0 and leaves *cursor unchanged
   (DECISIONS L, 2026-11-07; the iOS stub does the same). Without this early return the unsigned
   w_after - start below would wrap around and the cursor would jump back 15 s. */
static inline int hist_read(es_hist *h, uint64_t *cursor, float *dst, int max) {
  uint64_t keep = (uint64_t)h->size - ES_HIST_GUARD;      /* how far back a read may start */
  uint64_t w = atomic_load_explicit(&h->w, memory_order_acquire);
  if (*cursor > w) return 0;                               /* from the future: read nothing, keep cursor */
  uint64_t oldest = w > keep ? w - keep : 0;
  uint64_t start = *cursor > oldest ? *cursor : oldest;
  uint64_t avail = w - start;                              /* start <= w here */
  uint32_t n = max > 0 ? (uint32_t)max : 0;
  if (avail < n) n = (uint32_t)avail;

  uint32_t at = (uint32_t)(start & h->mask);
  uint32_t first = h->size - at;
  if (first > n) first = n;
  memcpy(dst, h->buf + at, sizeof(float) * first);
  memcpy(dst + first, h->buf, sizeof(float) * (n - first));

  atomic_thread_fence(memory_order_acquire);               /* the copy happens before the re-check */
  uint64_t w_after = atomic_load_explicit(&h->w, memory_order_relaxed);
  if (w_after - start > keep) {                            /* the writer may have overwritten the copy */
    *cursor = w_after - keep;
    return 0;
  }
  *cursor = start + n;
  return (int)n;
}

/* Reader: copy the newest n items (n <= size - ES_HIST_GUARD) into dst. Returns how many were copied
   (fewer if less has been written so far). Retries up to 3 times if the writer laps the copy. */
static inline int hist_read_last(es_hist *h, float *dst, int n) {
  for (int attempt = 0; attempt < 3; attempt++) {
    uint64_t w = atomic_load_explicit(&h->w, memory_order_acquire);
    uint64_t cursor = w > (uint64_t)n ? w - (uint64_t)n : 0;
    int got = hist_read(h, &cursor, dst, n);
    if (got > 0 || w == 0) return got;
  }
  return 0;
}

#endif /* EARSHOT_RING_H */
