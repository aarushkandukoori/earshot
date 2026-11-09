/* wav.h - a minimal WAV reader and writer for the host tools (esrun) and the unit tests
 * (20-engine 1 file plan, 14.7). Never compiled into the app.
 *
 * Reads PCM16, PCM24 and float32 files (also when wrapped in WAVE_FORMAT_EXTENSIBLE), any channel
 * count, and returns channel 0 as float32. Writes mono float32 or PCM16.
 * Threads: any non-RT thread (allocates, does file I/O).
 */
#ifndef ES_WAV_H
#define ES_WAV_H

#include <stddef.h>
#include <stdint.h>

/* Decode a whole WAV file held in memory. On success returns 0 and sets *pcm (malloc'd, the caller
   frees it), *n (frames) and *rate (Hz). On failure returns -1, leaves *pcm NULL and writes a reason
   into err (when err is not NULL). PCM16 is scaled by 1/32768, PCM24 by 1/8388608. */
int wav_decode(const uint8_t *buf, size_t len, float **pcm, int *n, int *rate, char *err, size_t errlen);

/* wav_decode on the contents of the file at path. */
int wav_read(const char *path, float **pcm, int *n, int *rate, char *err, size_t errlen);

/* Encode n mono samples as a WAV file in memory: float32 (pcm16 = 0) or PCM16 (pcm16 = 1: clipped to
   [-1, 1], scaled by 32767 and rounded; NaN becomes 0). Returns a malloc'd buffer and its size in *len,
   or NULL on bad arguments or no memory. */
uint8_t *wav_encode(const float *pcm, int n, int rate, int pcm16, size_t *len);

/* wav_encode written to the file at path. 0 on success, -1 with a reason in err on failure. */
int wav_write(const char *path, const float *pcm, int n, int rate, int pcm16, char *err, size_t errlen);

#endif /* ES_WAV_H */
