/* wav.c - a minimal WAV reader and writer (see wav.h). Host tools and tests only (20-engine 14.7).
 *
 * A WAV file is a RIFF container: the 12-byte header "RIFF" <size> "WAVE", then chunks, each
 * <4-byte id> <4-byte little-endian size> <size bytes> (plus one pad byte when size is odd).
 * We need two chunks:
 *   "fmt " - format tag (1 = integer PCM, 3 = IEEE float, 0xFFFE = extensible, whose real tag is the
 *            first 2 bytes of the sub-format GUID at offset 24), channels, sample rate, block align
 *            (bytes per frame) and bits per sample;
 *   "data" - the interleaved samples, frame after frame.
 * Every other chunk (LIST, fact, ...) is skipped. All numbers are little-endian; we read them byte by
 * byte, so the code does not depend on the host's byte order.
 * Threads: any non-RT thread (allocates, does file I/O).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wav.h"

static uint32_t rd16(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8); }
static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static void wr16(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void wr32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

static int fail(char *err, size_t errlen, const char *msg) {
  if (err && errlen) snprintf(err, errlen, "%s", msg);
  return -1;
}

enum { KIND_PCM16 = 1, KIND_PCM24, KIND_F32 };

int wav_decode(const uint8_t *buf, size_t len, float **pcm, int *n, int *rate, char *err, size_t errlen) {
  if (!pcm || !n || !rate) return fail(err, errlen, "bad arguments");
  *pcm = NULL;
  if (!buf || len < 12 || memcmp(buf, "RIFF", 4) != 0 || memcmp(buf + 8, "WAVE", 4) != 0)
    return fail(err, errlen, "not a RIFF/WAVE file");

  /* 1. Walk the chunks and remember fmt and data. */
  uint32_t tag = 0, channels = 0, sr = 0, align = 0, bits = 0;
  const uint8_t *data = NULL;
  size_t data_len = 0;
  int have_fmt = 0;
  size_t pos = 12;
  while (pos + 8 <= len) {
    const uint8_t *id = buf + pos;
    size_t size = rd32(buf + pos + 4), body = pos + 8, avail = len - body;
    if (memcmp(id, "fmt ", 4) == 0) {
      if (size < 16 || size > avail) return fail(err, errlen, "fmt chunk too short");
      const uint8_t *f = buf + body;
      tag = rd16(f);
      channels = rd16(f + 2);
      sr = rd32(f + 4);
      align = rd16(f + 12);
      bits = rd16(f + 14);
      if (tag == 0xFFFE) {                     /* WAVE_FORMAT_EXTENSIBLE: real tag in the GUID */
        if (size < 40) return fail(err, errlen, "extensible fmt chunk too short");
        tag = rd16(f + 24);
      }
      have_fmt = 1;
    } else if (memcmp(id, "data", 4) == 0) {
      data = buf + body;
      data_len = size <= avail ? size : avail; /* a streamed or cut file: use what is there */
    }
    if (size > avail) break;                   /* nothing can follow a chunk that runs past the end */
    pos = body + size + (size & 1);
  }
  if (!have_fmt) return fail(err, errlen, "no fmt chunk");
  if (!data) return fail(err, errlen, "no data chunk");

  /* 2. Check the format. */
  int kind = 0;
  if (tag == 1 && bits == 16) kind = KIND_PCM16;
  else if (tag == 1 && bits == 24) kind = KIND_PCM24;
  else if (tag == 3 && bits == 32) kind = KIND_F32;
  if (!kind) {
    if (err && errlen) snprintf(err, errlen, "unsupported format tag %u with %u bits", tag, bits);
    return -1;
  }
  if (channels < 1) return fail(err, errlen, "zero channels");
  if (sr < 1 || sr > 1000000) return fail(err, errlen, "bad sample rate");
  uint32_t bps = bits / 8;
  if (align != channels * bps) return fail(err, errlen, "block align does not match channels x bytes");
  size_t frames = data_len / align;
  if (frames > 0x7fffffff) return fail(err, errlen, "file too long");

  /* 3. Convert channel 0 of every frame to float. */
  float *out = malloc(sizeof(float) * (frames ? frames : 1));
  if (!out) return fail(err, errlen, "out of memory");
  for (size_t i = 0; i < frames; i++) {
    const uint8_t *s = data + i * align;       /* channel 0 is the first sample of the frame */
    if (kind == KIND_PCM16) {
      out[i] = (float)(int16_t)rd16(s) / 32768.0f;
    } else if (kind == KIND_PCM24) {
      int32_t v = (int32_t)((uint32_t)s[0] | ((uint32_t)s[1] << 8) | ((uint32_t)s[2] << 16));
      if (v & 0x800000) v -= 0x1000000;        /* sign-extend 24 bits */
      out[i] = (float)v / 8388608.0f;
    } else {
      uint32_t u = rd32(s);
      memcpy(&out[i], &u, 4);                  /* IEEE float bits, exact */
    }
  }
  *pcm = out;
  *n = (int)frames;
  *rate = (int)sr;
  return 0;
}

int wav_read(const char *path, float **pcm, int *n, int *rate, char *err, size_t errlen) {
  if (pcm) *pcm = NULL;
  FILE *f = path ? fopen(path, "rb") : NULL;
  if (!f) return fail(err, errlen, "cannot open file");
  uint8_t *buf = NULL;
  long len = -1;
  if (fseek(f, 0, SEEK_END) == 0) len = ftell(f);
  if (len >= 0 && fseek(f, 0, SEEK_SET) == 0) buf = malloc((size_t)len + 1);
  int ok = buf && fread(buf, 1, (size_t)len, f) == (size_t)len;
  fclose(f);
  if (!ok) {
    free(buf);
    return fail(err, errlen, "cannot read file");
  }
  int r = wav_decode(buf, (size_t)len, pcm, n, rate, err, errlen);
  free(buf);
  return r;
}

uint8_t *wav_encode(const float *pcm, int n, int rate, int pcm16, size_t *len) {
  if ((!pcm && n > 0) || n < 0 || rate < 1 || !len) return NULL;
  uint32_t bps = pcm16 ? 2 : 4;
  size_t data_bytes = (size_t)n * bps;
  if (data_bytes > 0xFFFFFFFFu - 36) return NULL;
  uint8_t *b = malloc(44 + data_bytes);
  if (!b) return NULL;
  memcpy(b, "RIFF", 4);
  wr32(b + 4, (uint32_t)(36 + data_bytes));
  memcpy(b + 8, "WAVE", 4);
  memcpy(b + 12, "fmt ", 4);
  wr32(b + 16, 16);                            /* fmt chunk size */
  wr16(b + 20, pcm16 ? 1 : 3);                 /* 1 = PCM, 3 = IEEE float */
  wr16(b + 22, 1);                             /* mono */
  wr32(b + 24, (uint32_t)rate);
  wr32(b + 28, (uint32_t)rate * bps);          /* bytes per second */
  wr16(b + 32, bps);                           /* block align */
  wr16(b + 34, bps * 8);                       /* bits per sample */
  memcpy(b + 36, "data", 4);
  wr32(b + 40, (uint32_t)data_bytes);
  uint8_t *d = b + 44;
  for (int i = 0; i < n; i++) {
    if (pcm16) {
      float v = pcm[i];
      if (!(v == v)) v = 0.0f;                 /* NaN -> 0 */
      v = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
      wr16(d + 2 * (size_t)i, (uint32_t)(uint16_t)(int16_t)lrintf(v * 32767.0f));
    } else {
      uint32_t u;
      memcpy(&u, &pcm[i], 4);
      wr32(d + 4 * (size_t)i, u);
    }
  }
  *len = 44 + data_bytes;
  return b;
}

int wav_write(const char *path, const float *pcm, int n, int rate, int pcm16, char *err, size_t errlen) {
  size_t len = 0;
  uint8_t *b = wav_encode(pcm, n, rate, pcm16, &len);
  if (!b) return fail(err, errlen, "cannot encode (bad arguments or out of memory)");
  FILE *f = path ? fopen(path, "wb") : NULL;
  int ok = f && fwrite(b, 1, len, f) == len;
  if (f && fclose(f) != 0) ok = 0;
  free(b);
  return ok ? 0 : fail(err, errlen, "cannot write file");
}
