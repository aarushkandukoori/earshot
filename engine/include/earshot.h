#ifndef EARSHOT_H
#define EARSHOT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define ES_API_VERSION      1
#define ES_SR               16000
#define ES_HOP              64        /* samples per hop at 16 kHz (4 ms) */
#define ES_NFFT             512
#define ES_NBINS            257
#define ES_NBANDS           16
#define ES_D                64
#define ES_WESPK_DIM        256
#define ES_G_DIM            128
#define ES_TOK_DIM          64
#define ES_MAX_TGT_TOKENS   48
#define ES_MAX_OWN_TOKENS   16
#define ES_NULL_TOKENS      4
#define ES_MAX_TOKENS       68
#define ES_MAX_INSTANCES    3
#define ES_ALGO_LATENCY     128       /* samples at 16 kHz */
#define ES_HISTORY_SAMPLES  262144    /* 16.4 s ring, 15.36 s readable (16,384-sample overrun guard) */
#define ES_PROBE_DBFS       (-30)     /* latency probe level; Phase 0 picks -30, -24 or -18 (DECISIONS row) */

typedef enum {
  ES_OK = 0, ES_ERR_ARG = -1, ES_ERR_FORMAT = -2, ES_ERR_VERSION = -3, ES_ERR_CONFIG = -4,
  ES_ERR_NOMEM = -5, ES_ERR_ENROLL_SHORT = -6, ES_ERR_BUSY = -7, ES_ERR_STATE = -8,
  ES_ERR_RATE = -9, ES_ERR_PROBE_WEAK = -10, ES_ERR_SELFTEST = -11, ES_ERR_THREAD = -12
} es_status;
typedef enum { ES_MODE_PERSON = 0, ES_MODE_EVERYONE = 1 } es_mode;
typedef enum { ES_ROLE_TARGET = 0, ES_ROLE_OWN = 1 } es_role;

const char *es_status_str(es_status s);                 /* static string, never NULL */

/* ---------- 1. Model weights (immutable after load, shareable) ---------- */
typedef struct es_model es_model;
es_status   es_model_load(const void *esw, size_t nbytes, es_model **out, char *err, size_t errlen);
void        es_model_free(es_model *m);
const char *es_model_config_hash(const es_model *m);    /* 64 hex chars, owned by m; architecture only */
const char *es_model_data_hash(const es_model *m);      /* data_sha256 = model_hash (3.1), 64 hex chars, owned by m */
int         es_model_latency_samples(const es_model *m);/* 128 */
es_status   es_selftest(const es_model *m, const void *est, size_t nbytes, float *max_err_out);

/* ---------- 2. Enrollment and conditioning (never on an RT thread) ---------- */
typedef struct es_voiceprint es_voiceprint;
es_voiceprint *es_voiceprint_new(void);
void        es_voiceprint_free(es_voiceprint *v);
int         es_enroll_num_frames(int n_samples);         /* 0 if n < 576 (so F starts at 2), else floor((n-64)/512)+1 */
void        es_vad(const float *pcm, int n, uint8_t *mask_out);  /* es_enroll_num_frames(n) entries; F = 0 writes nothing */
es_status   es_enroll(const es_model *m, const float *pcm, int n, const uint8_t *mask /* NULL = es_vad */,
                      es_role role, const float *e_wespk /* 256 or NULL */, es_voiceprint *out);
int         es_voiceprint_num_tokens(const es_voiceprint *v);
void        es_voiceprint_get_g(const es_voiceprint *v, float g_out[ES_G_DIM]);
int         es_voiceprint_has_wespk(const es_voiceprint *v);
float       es_voiceprint_speech_seconds(const es_voiceprint *v);   /* kept frames x 0.032 */
es_status   es_embed_g(const es_model *m, const float *pcm, int n, float g_out[ES_G_DIM]); /* "locked" check */

typedef struct es_cond es_cond;
es_cond    *es_cond_new(void);
void        es_cond_free(es_cond *c);
es_status   es_cond_build(const es_model *m, es_mode mode, const es_voiceprint *target /* NULL iff EVERYONE */,
                          const es_voiceprint *own /* NULL = none */, int use_wespk, es_cond *out);

/* ---------- 3. One stream = one instance (the model, hop by hop) ---------- */
typedef struct es_stream es_stream;
es_stream  *es_stream_new(const es_model *m);           /* allocates state and scratch */
void        es_stream_free(es_stream *s);
void        es_stream_reset(es_stream *s);               /* RT-safe: zero all state */
int         es_stream_hop(es_stream *s, const es_cond *c, const float in[ES_HOP],
                          float out[ES_HOP], float heads[2]);   /* RT-safe; returns 1 if a NaN reset happened */

/* ---------- 4. Engine = rings + resampling + worker + instances + post + safety ---------- */
typedef struct es_engine es_engine;
typedef struct {
  int io_rate;        /* 48000 or 44100, else ES_ERR_RATE */
  int io_frames;      /* granted callback size (frames), 16..4096 */
  int max_frames;     /* largest callback accepted, 4096 */
  int n_instances;    /* 1..3 */
  int slack16;        /* extra deadline slack in 16 kHz samples: 22, 43 or 64 */
} es_engine_config;
typedef struct {
  float voice_db;     /* -10..+6, added to the AGC target */
  float room_db;      /* -60 = off; -24, -15, -8 */
  float max_gain_db;  /* AGC ceiling 0..18; voice_db + max_gain_db is clamped to 24 */
  float out_gain_db;  /* final trim -40..0 */
  float slot_gain[ES_MAX_INSTANCES];  /* 0..1 per instance; changes crossfade over xfade_ms */
  float xfade_ms;     /* default 150 */
  int   duck_own;     /* 1 = -15 dB while p_self > 0.5 */
  int   gate;         /* 1 = activity gate on */
  int   agc;          /* 1 = slow voice AGC on */
  int   bypass;       /* 1 = play the delayed input instead of the model (A/B, diagnostics) */
} es_params;
typedef struct {
  uint64_t hops, underruns, nan_resets, probe_runs;
  int32_t  slack16, last_frames, howl;      /* howl = runaway guard active */
  float    p_target[ES_MAX_INSTANCES], p_self;
  float    in_rms_db, out_rms_db, agc_gain_db, limiter_gr_db;
  float    compute_us_last, compute_us_max;
} es_stats;
typedef struct { float loop_ms, spread_ms, psr_db; int n_ok; } es_probe_result;

void       es_params_default(es_params *p);  /* voice_db 0, room_db -60, max_gain_db 18, out_gain_db 0, slot_gain {1,0,0},
                                                xfade_ms 150, duck_own 1, gate 1, agc 1, bypass 0 (= Swift EngineParams) */
es_status  es_engine_new(const es_model *m, const es_engine_config *cfg, es_engine **out); /* starts muted (gain 0) */
void       es_engine_free(es_engine *e);                /* worker must be stopped */
void       es_engine_render(es_engine *e, const float *in, float *out, int n); /* RT: audio callback only */
int        es_engine_run_hops(es_engine *e);            /* RT: worker body, runs all complete hops */
es_status  es_engine_start_worker(es_engine *e, void *os_workgroup /* os_workgroup_t or NULL */);
void       es_engine_stop_worker(es_engine *e);
es_status  es_engine_set_cond(es_engine *e, int slot, const es_cond *c, int reset_state); /* may return BUSY;
             c == NULL empties the slot (zero output, no compute); the next non-NULL cond always resets (4.3) */
void       es_engine_set_params(es_engine *e, const es_params *p);
void       es_engine_set_mute(es_engine *e, int mute);   /* RT-safe; 5 ms ramp down; the first unmute and every
                                                            later one fade in over 400 ms (raised cosine) */
void       es_engine_cue(es_engine *e, int beeps);       /* RT-safe; 1..3 beeps of 60 ms at 880 Hz, -18 dBFS */
void       es_engine_get_stats(const es_engine *e, es_stats *s);   /* RT-safe snapshot */
float      es_engine_latency_ms(const es_engine *e);     /* Earshot's delay over a pass-through */
uint64_t   es_engine_input_position(const es_engine *e); /* RT-safe atomic load: samples written to the history */
int        es_engine_read_input(es_engine *e, uint64_t *cursor, float *dst, int max);  /* 16 kHz mic history from
             *cursor; a cursor older than the oldest held sample starts there; skipped = new - old cursor - returned */
int        es_engine_read_output(es_engine *e, int slot, float *dst, int n);  /* last n <= 48000 samples */
int        es_engine_read_timings(es_engine *e, uint16_t *us, int max);      /* per-hop compute, microseconds */
es_status  es_engine_probe_start(es_engine *e);
es_status  es_engine_probe_result(es_engine *e, es_probe_result *r);  /* STATE until done */

/* ---------- 5. DSP chain blocks (used inside render; public for tests) ---------- */
typedef struct { float gain, start, target, step, phase, sr; int shape; } es_ramp;  /* shape 0 linear, 1 raised cosine;
   gain = start + (target - start) * w(phase), phase 0..1, w = phase or 0.5 - 0.5 cos(pi phase) */
void es_ramp_init(es_ramp *r, float sr, float gain);
void es_ramp_to(es_ramp *r, float target, float ms, int shape);
void es_ramp_process(es_ramp *r, float *buf, int n);
typedef struct { float ceiling, rel_coef, gain, min_gain; } es_limiter;
void es_limiter_init(es_limiter *l, float sr, float ceiling_dbfs /* -3 */, float release_ms /* 50 */);
void es_limiter_process(es_limiter *l, float *buf, int n);   /* zero lookahead, instant attack */
void es_softclip(float *buf, int n);                        /* knee at -1 dBFS, output within [-1, 1] */

#ifdef __cplusplus
}
#endif
#endif
