/*
 * AudioHost.h - the RemoteIO audio unit and its C render callback (30-ios 4.2,
 * contract 5). Owned by WP-IOS.
 *
 * Swift creates one eh_host per RemoteIO lifetime. The render callback runs on
 * the audio thread and calls only AudioUnitRender, es_engine_render, memset and
 * memcpy. Swift never runs on that thread.
 */
#ifndef EARSHOT_AUDIOHOST_H
#define EARSHOT_AUDIOHOST_H

#include <AudioToolbox/AudioToolbox.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EH_MAX_FRAMES 4096 /* kAudioUnitProperty_MaximumFramesPerSlice, matches es_engine_config.max_frames */

typedef struct eh_host eh_host;

/* RemoteIO, input bus 1, Float32 mono mic; output bus 0, Float32 stereo (mono copied to both).
 * es_engine is the es_engine* the callback renders through. */
OSStatus eh_create(void *es_engine, double io_rate, int io_frames, eh_host **out);
OSStatus eh_start(eh_host *h);
OSStatus eh_stop(eh_host *h);

/* kAudioOutputUnitProperty_OSWorkgroup, size checked, NULL on failure. The host keeps
 * ownership (it releases the workgroup in eh_destroy). */
void *eh_workgroup(eh_host *h);

/* Atomic swap of the engine the callback uses. Returns once the callback acknowledged it
 * (or at once if I/O is stopped), and an error after 200 ms without an acknowledge (then the
 * old engine is still the one in use and must not be freed). */
OSStatus eh_set_engine(eh_host *h, void *es_engine);

/* Debug builds (EARSHOT_MALLOC_HOOK): the id of the thread that ran the newest render callback,
 * for the debug malloc counter (MallocHook.h). 0 before the first callback and always 0 in Release. */
uint64_t eh_render_thread_id(eh_host *h);

void eh_destroy(eh_host *h);

#ifdef __cplusplus
}
#endif

#endif /* EARSHOT_AUDIOHOST_H */
