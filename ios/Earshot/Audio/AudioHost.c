/*
 * AudioHost.c - RemoteIO plus the C render callback (30-ios 4.2, contract 5).
 *
 * Picture of one audio callback (every ~1.3 ms at 64 frames, 48 kHz):
 *
 *   iOS asks bus 0 for n output frames -> eh_render():
 *     1. AudioUnitRender(bus 1) pulls n mic frames into in_buf
 *     2. es_engine_render(engine, in_buf, left, n)  (all DSP, mute and limiter live there)
 *     3. memcpy(left -> right), so both AirPods play
 *
 * Real-time rules for eh_render (checked in every review of Audio/): no malloc/free,
 * locks, dispatch, Swift, Objective-C, printf/os_log, file I/O or AudioUnitSetProperty.
 * The only calls are AudioUnitRender, es_engine_render, memset and memcpy. The engine
 * swap below uses one lock-free atomic exchange, which is a CPU instruction, not a call.
 *
 * Debug builds only (EARSHOT_MALLOC_HOOK, I-1): the callback also stores its own thread id
 * (pthread_threadid_np on the calling thread is a field read, no lock, no syscall) so the
 * debug malloc counter (MallocHook.c) can be armed for exactly this thread. Release builds
 * compile that line out, so the rule above holds word for word there.
 */
#include "AudioHost.h"

#include <os/object.h>
#include <pthread.h>
#include <os/workgroup.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "earshot.h"

#define EH_ERR_TIMEOUT ((OSStatus)-1) /* eh_set_engine: no acknowledge within 200 ms */

struct eh_host {
  AudioComponentInstance unit; /* the RemoteIO unit */
  es_engine *engine;           /* engine the callback renders through (callback-owned while running) */
  _Atomic(void *) requested;   /* a new engine waiting to be picked up, NULL = none */
  float *in_buf;               /* EH_MAX_FRAMES mic samples, allocated once in eh_create */
  AudioBufferList abl;         /* one mono buffer pointing at in_buf */
  os_workgroup_t workgroup;    /* +1 reference from AudioUnitGetProperty, or NULL */
  _Atomic uint64_t render_tid; /* Debug only: id of the thread that runs eh_render, 0 until the first callback */
};

/* The render callback. Runs on the audio I/O thread. */
static OSStatus eh_render(void *ref, AudioUnitRenderActionFlags *flags, const AudioTimeStamp *ts, UInt32 bus,
                          UInt32 n, AudioBufferList *io) {
  (void)bus;
  eh_host *h = (eh_host *)ref;
  float *left = (float *)io->mBuffers[0].mData;
#ifdef EARSHOT_MALLOC_HOOK
  uint64_t tid = 0; /* Debug only: tells the malloc counter which thread renders */
  pthread_threadid_np(NULL, &tid);
  atomic_store_explicit(&h->render_tid, tid, memory_order_relaxed);
#endif

  /* Engine swap: if Swift asked for a new engine, take it now. Writing NULL back is
   * the acknowledge that eh_set_engine waits for. */
  void *next = atomic_exchange_explicit(&h->requested, NULL, memory_order_acq_rel);
  if (next != NULL) h->engine = (es_engine *)next;

  if (n > EH_MAX_FRAMES || h->engine == NULL) { /* cannot happen with MaximumFramesPerSlice 4096 */
    for (UInt32 b = 0; b < io->mNumberBuffers; b++) memset(io->mBuffers[b].mData, 0, n * sizeof(float));
    return noErr;
  }
  h->abl.mBuffers[0].mData = h->in_buf;
  h->abl.mBuffers[0].mDataByteSize = n * sizeof(float);
  if (AudioUnitRender(h->unit, flags, ts, 1, n, &h->abl) != noErr)
    memset(h->in_buf, 0, n * sizeof(float)); /* mic not ready yet: process silence */
  es_engine_render(h->engine, h->in_buf, left, (int)n); /* the only engine call on this thread */
  if (io->mNumberBuffers > 1) memcpy(io->mBuffers[1].mData, left, n * sizeof(float));
  return noErr;
}

/* Linear PCM Float32, packed, non-interleaved, at the hardware rate. */
static AudioStreamBasicDescription eh_float_format(double rate, UInt32 channels) {
  AudioStreamBasicDescription f;
  memset(&f, 0, sizeof(f));
  f.mSampleRate = rate;
  f.mFormatID = kAudioFormatLinearPCM;
  f.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved;
  f.mBytesPerPacket = sizeof(float); /* non-interleaved: sizes are per channel */
  f.mFramesPerPacket = 1;
  f.mBytesPerFrame = sizeof(float);
  f.mChannelsPerFrame = channels;
  f.mBitsPerChannel = 32;
  return f;
}

OSStatus eh_create(void *es_engine_ptr, double io_rate, int io_frames, eh_host **out) {
  if (out == NULL || es_engine_ptr == NULL || io_frames < 16 || io_frames > EH_MAX_FRAMES) return kAudio_ParamError;
  *out = NULL;
  eh_host *h = (eh_host *)calloc(1, sizeof(eh_host));
  if (h == NULL) return kAudio_MemFullError;
  h->engine = (es_engine *)es_engine_ptr;
  atomic_init(&h->requested, NULL);
  atomic_init(&h->render_tid, 0);

  /* 1. Find and instantiate RemoteIO. */
  AudioComponentDescription desc;
  memset(&desc, 0, sizeof(desc));
  desc.componentType = kAudioUnitType_Output;
  desc.componentSubType = kAudioUnitSubType_RemoteIO;
  desc.componentManufacturer = kAudioUnitManufacturer_Apple;
  AudioComponent comp = AudioComponentFindNext(NULL, &desc);
  if (comp == NULL) {
    free(h);
    return kAudioUnitErr_InvalidElement;
  }
  OSStatus st = AudioComponentInstanceNew(comp, &h->unit);
  if (st != noErr) {
    free(h);
    return st;
  }

  /* 2. Enable the mic (input scope, bus 1) and the speaker side (output scope, bus 0). */
  UInt32 one = 1;
  st = AudioUnitSetProperty(h->unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, &one, sizeof(one));
  if (st == noErr)
    st = AudioUnitSetProperty(h->unit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &one,
                              sizeof(one));

  /* 3. Formats at the hardware rate: mic side mono, speaker side 2 channels. */
  AudioStreamBasicDescription mic = eh_float_format(io_rate, 1);
  AudioStreamBasicDescription spk = eh_float_format(io_rate, 2);
  if (st == noErr)
    st = AudioUnitSetProperty(h->unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, &mic, sizeof(mic));
  if (st == noErr)
    st = AudioUnitSetProperty(h->unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &spk, sizeof(spk));

  /* 4. Largest callback we accept. */
  UInt32 max_frames = EH_MAX_FRAMES;
  if (st == noErr)
    st = AudioUnitSetProperty(h->unit, kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, 0,
                              &max_frames, sizeof(max_frames));

  /* 5. Render callback on bus 0, plus the mic buffer (allocated here, never later). */
  AURenderCallbackStruct cb;
  cb.inputProc = eh_render;
  cb.inputProcRefCon = h;
  if (st == noErr)
    st = AudioUnitSetProperty(h->unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &cb,
                              sizeof(cb));
  h->in_buf = (float *)calloc(EH_MAX_FRAMES, sizeof(float));
  if (st == noErr && h->in_buf == NULL) st = kAudio_MemFullError;
  h->abl.mNumberBuffers = 1;
  h->abl.mBuffers[0].mNumberChannels = 1;
  h->abl.mBuffers[0].mData = h->in_buf;
  h->abl.mBuffers[0].mDataByteSize = EH_MAX_FRAMES * sizeof(float);

  /* 6. Initialize, then read the I/O workgroup once and keep it only if it looks right. */
  if (st == noErr) st = AudioUnitInitialize(h->unit);
  if (st == noErr) {
    os_workgroup_t wg = NULL;
    UInt32 sz = sizeof(wg);
    OSStatus ws =
        AudioUnitGetProperty(h->unit, kAudioOutputUnitProperty_OSWorkgroup, kAudioUnitScope_Global, 0, &wg, &sz);
    if (ws == noErr && sz == sizeof(wg) && wg != NULL) {
      h->workgroup = wg; /* +1 reference (SDK header comment), released in eh_destroy */
    }
    /* Otherwise keep NULL and never call os_release: with a wrong size the value may be only
       partly written, so it is not a real object (30-ios 4.2). At worst one reference leaks. */
  }

  if (st != noErr) {
    eh_destroy(h);
    return st;
  }
  *out = h;
  return noErr;
}

OSStatus eh_start(eh_host *h) {
  if (h == NULL) return kAudio_ParamError;
  return AudioOutputUnitStart(h->unit);
}

OSStatus eh_stop(eh_host *h) {
  if (h == NULL) return kAudio_ParamError;
  return AudioOutputUnitStop(h->unit);
}

void *eh_workgroup(eh_host *h) {
  if (h == NULL) return NULL;
  return (void *)h->workgroup;
}

/* Is RemoteIO currently pulling callbacks? */
static int eh_is_running(eh_host *h) {
  UInt32 running = 0;
  UInt32 sz = sizeof(running);
  OSStatus st =
      AudioUnitGetProperty(h->unit, kAudioOutputUnitProperty_IsRunning, kAudioUnitScope_Global, 0, &running, &sz);
  return st == noErr && running != 0;
}

OSStatus eh_set_engine(eh_host *h, void *es_engine_ptr) {
  if (h == NULL || es_engine_ptr == NULL) return kAudio_ParamError;
  if (!eh_is_running(h)) { /* no callbacks: swap directly */
    atomic_store_explicit(&h->requested, NULL, memory_order_release);
    h->engine = (es_engine *)es_engine_ptr;
    return noErr;
  }
  atomic_store_explicit(&h->requested, es_engine_ptr, memory_order_release);
  for (int i = 0; i < 200; i++) { /* poll every 1 ms for up to 200 ms */
    if (atomic_load_explicit(&h->requested, memory_order_acquire) == NULL) return noErr; /* acknowledged */
    usleep(1000);
  }
  /* Timeout: withdraw the request. If the exchange fails, the callback took it just now. */
  void *expected = es_engine_ptr;
  if (atomic_compare_exchange_strong_explicit(&h->requested, &expected, NULL, memory_order_acq_rel,
                                              memory_order_acquire))
    return EH_ERR_TIMEOUT; /* withdrawn: the old engine is still in use */
  return noErr;
}

uint64_t eh_render_thread_id(eh_host *h) {
  if (h == NULL) return 0;
  return atomic_load_explicit(&h->render_tid, memory_order_relaxed); /* stays 0 in Release */
}

void eh_destroy(eh_host *h) {
  if (h == NULL) return;
  if (h->unit != NULL) {
    AudioOutputUnitStop(h->unit);
    AudioUnitUninitialize(h->unit);
    AudioComponentInstanceDispose(h->unit);
  }
  if (h->workgroup != NULL) os_release(h->workgroup);
  free(h->in_buf);
  free(h);
}
