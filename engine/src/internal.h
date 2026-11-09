/* internal.h - private declarations shared by the engine's .c files and its tests.
 *
 * Never included by Swift or by AudioHost.c: only include/earshot.h is public (20-engine 2.3).
 * Implements the private side of contract sections 2 to 4 and 20-engine sections 3 to 11.
 *
 * What lives here
 *   - small numeric constants (band layout, window sizes)
 *   - the JSON tokenizer types (json_min.c) and the container reader types (model.c, 20-engine 4.1)
 *   - the model, conditioning and voiceprint structs (20-engine 4.3, 7.3, 7.4), the 270-tensor list
 *     and the synthetic test model (contract 3.2, 20-engine 4.4)
 *   - the FFT wrapper and STFT helpers (stft.c, 20-engine 5)
 *   - the neural-network kernels (nn.c, 20-engine 3.3 and 6.2)
 *   - the polyphase resampler (resample.c, 20-engine 9)
 *   - the energy VAD helpers live in earshot.h (es_vad, es_enroll_num_frames), enroll.c implements them
 *
 * Package E-0 built this file. E-1 (model load, stream hop, enrollment, conditioning) and E-2 (pipeline,
 * worker, post, safety chain) add their structs (es_stream, es_engine) and prototypes at the marked places.
 */
#ifndef EARSHOT_INTERNAL_H
#define EARSHOT_INTERNAL_H

#include <stddef.h>
#include <stdint.h>
#include <Accelerate/Accelerate.h>

#include "earshot.h"

/* ------------------------------------------------------------------------------------------------
 * Constants (contract 2.1 and 2.2). They repeat the canonical config of 20-engine 4.2.
 * ---------------------------------------------------------------------------------------------- */
#define ES_PI          3.14159265358979323846   /* our own pi: strict C11 does not promise M_PI */
#define ES_ANA_RISE    448                      /* analysis window rise length (samples) */
#define ES_SYNTH_LEN   128                      /* synthesis window length (samples) */
#define ES_SYNTH_START 384                      /* ES_NFFT - ES_SYNTH_LEN: first non-zero ws index */
#define ES_NFEAT       (3 * ES_NBINS)           /* 771 features per frame, index 3*bin + f */
#define ES_DF_BINS     128                      /* bins 0..127 use the 3-tap deep filter */
#define ES_DF_TAPS     3
#define ES_COEF_TOTAL  1026                     /* sum over bands of 2 * W_k * T_k */
#define ES_LN_EPS      1e-5f
#define ES_TOK_HOP     512                      /* token frames hop (contract 2.4) */

/* ------------------------------------------------------------------------------------------------
 * JSON tokenizer (json_min.c, 20-engine 4.1).
 * A flat token array in document order: a parent always comes before its children.
 * For an object, the children are its keys (string tokens); each key has exactly one child, its value.
 * ---------------------------------------------------------------------------------------------- */
typedef enum {
  ES_JSON_OBJECT = 1, ES_JSON_ARRAY, ES_JSON_STRING, ES_JSON_NUMBER, ES_JSON_TRUE, ES_JSON_FALSE, ES_JSON_NULL
} es_jtype;

typedef struct {
  int type;     /* es_jtype */
  int start;    /* byte offset of the first character (for strings: the first byte after the opening quote) */
  int end;      /* byte offset one past the last character (for strings: the closing quote) */
  int size;     /* number of direct children (object: keys, array: elements, key: 1, others: 0) */
  int parent;   /* index of the parent token, -1 for the root */
  int next;     /* index of the first token after this token's whole subtree (used to skip children) */
} es_jtok;

typedef struct {
  const char *js;   /* the JSON text (not NUL-terminated) */
  uint32_t len;     /* its length in bytes */
  es_jtok *tok;     /* malloc'd token array (one malloc per document) */
  int n_tok;        /* tokens in use; token 0 is the root value */
} es_json;

#define ES_JSON_MAX_DEPTH 32   /* deeper nesting is rejected (the real headers nest 4 levels) */
#define ES_JSON_MAX_NUM   64   /* number tokens longer than this are rejected (20-engine 4.1) */

es_status es_json_parse(es_json *j, const char *js, uint32_t len);  /* ES_OK or ES_ERR_FORMAT/NOMEM */
void      es_json_free(es_json *j);                                 /* frees the token array */
int       es_json_get(const es_json *j, int obj, const char *key);  /* value token of key, or -1 */
int       es_json_get_path(const es_json *j, int obj, const char *dotted);  /* "meta.tolerances.out" */
int       es_json_arr_at(const es_json *j, int arr, int i);         /* element i, or -1 */
int       es_json_num(const es_json *j, int t, double *out);        /* 0 = ok (finite number), -1 = not */
int       es_json_str_eq(const es_json *j, int t, const char *s);   /* 1 if string token t equals s */
int       es_json_str_len(const es_json *j, int t);                 /* raw byte length of a string, or -1 */

/* ------------------------------------------------------------------------------------------------
 * Container reader for .esw (weights) and .est (test vectors), model.c, 20-engine 4.1, contract 3.1.
 * ---------------------------------------------------------------------------------------------- */
enum { ES_DT_F32 = 0, ES_DT_U8 = 1, ES_DT_I32 = 2 };

typedef struct {
  const char *name;     /* NUL-terminated copy owned by the container */
  int dtype;            /* ES_DT_* */
  int ndim;             /* 0..4 */
  int shape[4];
  const void *data;     /* points into the caller's buffer: base + data_start + offset */
  size_t nbytes;
  size_t offset;        /* relative to data_start, a multiple of 64 */
} es_array;

typedef struct {
  const uint8_t *base;  /* the caller's file buffer (never freed by the container) */
  size_t n;             /* its size in bytes */
  es_json j;            /* the parsed JSON header; token 0 is the header object */
  size_t data_start;    /* round_up(16 + header_len, 64) */
  size_t data_bytes;    /* n - data_start */
  es_array *arrays;     /* the "arrays" list, in file order */
  int n_arrays;
  char *names;          /* storage for the NUL-terminated array names */
  char config_hash[65]; /* verified config_hash (lowercase hex) */
  char data_hash[65];   /* verified data_sha256 (lowercase hex) = model_hash for a .esw */
} es_container;

es_status es_container_open(const void *buf, size_t n, const char magic[4], es_container *ct,
                            char *err, size_t errlen);
const es_array *es_container_find(const es_container *ct, const char *name);
void es_container_close(es_container *ct);   /* frees arrays, names and tokens, never the input buffer */

/* The canonical config string and its hash (20-engine 4.2). Defined in model.c. */
extern const char ES_CONFIG_CANON[];
#define ES_CONFIG_HASH_HEX "089d72c06d244e1650d209453c91ddffde990b9589dd7a9c228f9e1ea3b93aec"
void es_sha256_hex(const void *data, size_t n, char out[65]);   /* CommonCrypto SHA-256 as lowercase hex */

/* ------------------------------------------------------------------------------------------------
 * Model, conditioning and voiceprint structs (20-engine 4.3, 7.3, 7.4).
 * E-0 declares them and builds the synthetic es_model (model.c); E-1 loads es_model from a file and
 * fills es_cond and es_voiceprint in model.c, enroll.c and cond.c.
 * ---------------------------------------------------------------------------------------------- */
typedef struct { const float *w, *b; } es_lin;                /* w [in][out] row-major, b [out] (may be NULL) */
typedef struct { const float *w, *b; } es_ln;                 /* LayerNorm weight and bias, 64 each */
typedef struct { const float *wih, *whh, *bias; } es_lstmw;  /* [in][4H], [H][4H], [4H]; gates i f g o */

struct es_model {
  void *arena; size_t arena_bytes; char config_hash[65], data_hash[65];
  int band_start[16], band_width[16], band_taps[16], coef_off[16];
  float wa[512], ws[512];
  es_lin enc_lin[16]; es_ln enc_ln[16];
  struct { es_ln tn, bn; es_lstmw t, b, b_rev; es_lin tf, bf; } blk[4];
  es_lin att_q, att_o, att_kv; const float *type_emb, *null_tok;           /* [3][64], [4][64] */
  es_lin film0, film2; const float *null_e_t, *null_g_t, *null_e_o, *null_g_o;
  struct { es_ln ln; es_lin fc1, fc2; } dec[16];
  es_lin heads;
  struct { es_ln tn, bn; es_lstmw t, t_rev, b, b_rev; es_lin tf, bf; } tok;
  es_lin tok_proj, tok_g;
};

/* Conditioning set: 12,800 floats + 2 ints. n_tokens == 0 marks an empty engine slot (20-engine 8.5). */
struct es_cond {
  _Alignas(64) float film[4096];
  float k[4][68][16];
  float v[4][68][16];
  int32_t n_tokens;
  int32_t mode;
};

/* Voiceprint of one enrolled person (20-engine 7.3), fixed size. */
struct es_voiceprint {
  float tokens[ES_MAX_TGT_TOKENS][ES_TOK_DIM];
  int n_tok;
  float g[ES_G_DIM];
  float e[ES_WESPK_DIM];
  int has_wespk;
  int role;
  float speech_s;
};

/* Debug taps for the parity test (20-engine 6.6). Any pointer may be NULL. */
typedef struct {
  float *feat;       /* [771] after the features */
  float *z_enc;      /* [16][64] after the band encoder */
  float *z_blk[4];   /* [16][64] at the end of each block, including FiLM and attention */
  float *coef;       /* [1026] after 1.5 tanh */
} es_dbg;

/* The 270 weight tensors of contract 3.2 in file order (model.c). The synthetic model (20-engine 4.4)
   fills each one; es_model_load (E-1, 4.3) binds each one by name. A tensor's shape is [d0] when
   d1 == 0, else [d0, d1]; for a 2-D weight d0 is the fan-in (Linear [in, out], LSTM [in or H, 4H]). */
enum {
  ES_INIT_WEIGHT = 0,   /* synthetic value u / sqrt(d0), u uniform in [-1, 1) */
  ES_INIT_BIAS,         /* 0.1 u (Linear, LSTM and LayerNorm biases) */
  ES_INIT_LN_WEIGHT,    /* 1 + 0.1 u (LayerNorm weights) */
  ES_INIT_EMBED         /* 0.1 u (att.type_emb, att.null_tok and the null.* vectors) */
};
#define ES_N_TENSORS       270
#define ES_N_WEIGHT_FLOATS 2570500   /* contract 3.2: 2,570,500 floats in all */

typedef struct {
  char name[32];        /* e.g. "blocks.2.b.w_hh_rev" */
  int d0, d1;           /* shape */
  int init;             /* ES_INIT_* (used by the synthetic model only) */
  const float **dst;    /* the es_model field that must point at this tensor */
} es_tensor;

/* Fill list[ES_N_TENSORS] for model m in contract 3.2 order. Returns the count (always 270). */
int es_model_tensors(es_model *m, es_tensor *list);

/* Test-only model with random weights from xorshift32(seed) (20-engine 4.4). allpass = 1 makes every
   deep-filter coefficient 1 on tap 0 and 0 elsewhere, so the model output equals its input. Free it
   with es_model_free. ES_OK, ES_ERR_ARG (out is NULL) or ES_ERR_NOMEM. */
es_status es_model_synthetic(uint32_t seed, int allpass, es_model **out);

/* E-1 declarations (stream, token encoder, file loading, test-vector helpers) are at the end of this
   file, after the FFT wrapper they use. */

/* ------------------------------------------------------------------------------------------------
 * FFT wrapper and STFT helpers (stft.c, 20-engine 5).
 * Each user (stream, enrollment call, test) owns its own es_fft; never share one between threads.
 * ---------------------------------------------------------------------------------------------- */
typedef struct {
  FFTSetup setup;              /* vDSP_create_fftsetup(9, kFFTRadix2) */
  _Alignas(16) float re[256];  /* split-complex work buffers */
  _Alignas(16) float im[256];
} es_fft;

void      es_make_windows(float wa[ES_NFFT], float ws[ES_NFFT]);          /* 5.1, double math, float store */
es_status es_fft_init(es_fft *f);                                          /* ES_OK or ES_ERR_NOMEM */
void      es_fft_free(es_fft *f);
void      es_rfft(es_fft *f, const float x[ES_NFFT], float Xr[ES_NBINS], float Xi[ES_NBINS]);   /* = torch rfft */
void      es_irfft(es_fft *f, const float Yr[ES_NBINS], const float Yi[ES_NBINS], float y[ES_NFFT]); /* = irfft */
void      es_features(const float Xr[ES_NBINS], const float Xi[ES_NBINS], float feat[ES_NFEAT],
                      float work[3 * ES_NBINS]);                           /* 5.3; work = p, mag, t */
/* Steps 1 to 3 of a hop (20-engine 6.3): shift 64 samples into ana, window into frame, FFT. */
void      es_analysis_hop(es_fft *f, const float wa[ES_NFFT], float ana[ES_NFFT], const float in[ES_HOP],
                          float frame[ES_NFFT], float Xr[ES_NBINS], float Xi[ES_NBINS]);
/* Streaming synthesis (5.4): irfft, synthesis window, overlap-add into ola[128], emit 64 samples. */
void      es_synthesis_hop(es_fft *f, const float ws[ES_NFFT], const float Yr[ES_NBINS],
                           const float Yi[ES_NBINS], float frame[ES_NFFT], float ola[ES_SYNTH_LEN],
                           float out[ES_HOP]);
/* Offline STFT and iSTFT (contract 2.1), test-only. n = 64 * T samples, X arrays are [T][257]. */
es_status es_stft_offline(const float wa[ES_NFFT], const float *x, int n, float *Xr, float *Xi);
es_status es_istft_offline(const float ws[ES_NFFT], const float *Xr, const float *Xi, int T, float *y);

/* ------------------------------------------------------------------------------------------------
 * Neural-network kernels (nn.c, 20-engine 3.3 and 6.2). All float32, all RT-safe (no allocation).
 * ---------------------------------------------------------------------------------------------- */
/* Y[M][out] = X[M][in] * W[in][out] + b. b may be NULL (then Y = X * W). Y must not alias X. */
void es_linear(const float *X, int M, int in, int out, const float *W, const float *b, float *Y);
/* Y[M][out] += X[M][in] * W[in][out] (accumulate, no bias). Y must not alias X. */
void es_linear_acc(const float *X, int M, int in, int out, const float *W, float *Y);
/* The same two products for the real-time hop: one cblas_sgemv per row, never cblas_sgemm, because
   Accelerate's sgemm mallocs inside every call with M >= 2 on the M4 (nn.c explains; contract 4.1). */
void es_linear_rt(const float *X, int M, int in, int out, const float *W, const float *b, float *Y);
void es_linear_acc_rt(const float *X, int M, int in, int out, const float *W, float *Y);
/* The faster path (E-3, opt-in): cblas_sgemm on X transposed, in column blocks of at most 256, about
   3x less time than one sgemv per row. 0 allocations on macOS and the iOS 26.0 simulator; the iPhone is
   not measured yet (nn.c explains). xt is scratch of at least M * in floats, 64-byte aligned. */
void es_linear_xt(const float *X, int M, int in, int out, const float *W, const float *b, float *Y, float *xt);
void es_linear_acc_xt(const float *X, int M, int in, int out, const float *W, float *Y, float *xt);
/* The hop's 16-row products: es_linear_rt by default (what the app ships), es_linear_xt when built with
   -DES_HOP_SGEMM_XT (and without -DES_HOP_SGEMV, which always wins). es_hop_path names the compiled
   choice, "sgemv" or "xt" (bench rows and test output print it). */
void es_linear_hop(const float *X, int M, int in, int out, const float *W, const float *b, float *Y, float *xt);
void es_linear_acc_hop(const float *X, int M, int in, int out, const float *W, float *Y, float *xt);
extern const char *const es_hop_path;
/* Y[m][j] += b[j] for every row m (adds one bias row to M rows). */
void es_add_rows(float *Y, int M, int n, const float *b);
/* LayerNorm over C = 64 for R rows; X may equal Y. eps 1e-5. */
void es_layernorm(const float *X, int R, const es_ln *ln, float *Y);
/* R independent LSTM cells, hidden H. g = R x 4H preactivations (i, f, g, o), overwritten.
   h, c = R x H, updated in place. th = R x H scratch. */
void es_lstm_cell(int R, int H, float *g, float *h, float *c, float *th);
/* B independent LSTM sequences of length T from zero state (20-engine 6.2). work holds B * 7 * H floats. */
void es_lstm_seq(const float *gx, int gx_t, int gx_b, float *hs, int hs_t, int hs_b,
                 int T, int B, int H, const float *whh, int reverse, float *work);

/* ------------------------------------------------------------------------------------------------
 * Rational polyphase resampler (resample.c, resampler_tables.c, 20-engine 9).
 * ---------------------------------------------------------------------------------------------- */
/* One polyphase table: up L, down M, Q taps per phase; rev[p * Q + i] holds L * h[p + (Q-1-i) L]
   (each phase stored reversed so one output is one dot product over ascending memory). */
typedef struct { int L, M, Q; const float *rev; } es_rs_table;

/* Generated by tools/make_resampler_tables.py into resampler_tables.c. Index 0 = 48 kHz, 1 = 44.1 kHz. */
extern const es_rs_table es_rs_decim_tables[2];    /* I/O rate -> 16 kHz */
extern const es_rs_table es_rs_interp_tables[2];   /* 16 kHz -> I/O rate */
extern const float ES_RS_DELAY_MS[2];              /* decimator + interpolator mean group delay, 1 to 4 kHz */

#define ES_RS_MAX_DECIM_IN  4096   /* most I/O-rate samples per es_rs_process call on the decimator */
#define ES_RS_MAX_INTERP_IN 1600   /* most 16 kHz samples per call on the interpolator */

typedef struct {
  int L, M, Q;          /* copied from the table */
  const float *rev;     /* the table's reversed phases */
  float *hist;          /* Q old samples followed by the new block (caller allocates, see es_rs_hist_len) */
  int64_t base;         /* absolute index of the newest input sample the next output needs */
  int64_t consumed;     /* input samples fed so far */
  int p;                /* phase of the next output, 0..L-1 */
} es_rs;

int  es_rs_family(int io_rate);                       /* 0 for 48000, 1 for 44100, -1 otherwise */
int  es_rs_hist_len(const es_rs_table *t, int max_in);/* floats the caller must allocate for hist */
void es_rs_init(es_rs *r, const es_rs_table *t, float *hist);   /* zero history, base = consumed = p = 0 */
void es_rs_reset(es_rs *r);                           /* back to the init state, keeps table and hist */
/* Feed n_in samples (at most the max_in the hist was sized for), write at most max_out outputs. RT-safe. */
int  es_rs_process(es_rs *r, const float *in, int n_in, float *out, int max_out);
/* New input samples needed before n_out more outputs can be produced. */
int  es_rs_need(const es_rs *r, int n_out);
/* Count-only twin of es_rs_process (same stepping, no samples): returns the outputs it would make. */
int  es_rs_step_count(es_rs *r, int n_in, int max_out);
/* P_min of 20-engine 9.4 for (io_rate, io_frames); -1 for a bad rate or size. Non-RT (about 1 ms). */
int  es_rs_prefill_min(int io_rate, int io_frames);

/* ------------------------------------------------------------------------------------------------
 * Package E-1 (model.c, stream.c, enroll.c, cond.c, selftest.c).
 * ---------------------------------------------------------------------------------------------- */
/* Read a whole file into a malloc'd buffer (non-RT helper for tests, tools and es_model_load_file).
   ES_OK, ES_ERR_ARG (cannot open or read) or ES_ERR_NOMEM. The caller frees *buf. */
es_status es_read_file(const char *path, void **buf, size_t *n);
/* es_model_load on the contents of a file (the file buffer is freed before returning). */
es_status es_model_load_file(const char *path, es_model **out, char *err, size_t errlen);

/* ------------------------------------------------------------------------------------------------
 * Package E-1: one stream instance (stream.c, 20-engine 6.1). Allocated once by es_stream_new with
 * posix_memalign (64-byte aligned). es_stream_reset zeroes only `st`, the state of contract 2.6;
 * everything else is scratch that every hop overwrites before reading.
 * ---------------------------------------------------------------------------------------------- */
typedef struct {
  _Alignas(64) float ana[ES_NFFT];        /* analysis history, newest sample last */
  float xhist[2][ES_DF_BINS][2];          /* X of the previous 2 hops, bins 0..127, (re, im); 0 = t-1 */
  float h_t[4][16][128];                  /* time-LSTM hidden state per block per band */
  float c_t[4][16][128];                  /* time-LSTM cell state per block per band */
  float ola[ES_SYNTH_LEN];                /* overlap-add buffer */
} es_stream_state;                        /* 17,536 floats (70 KB) */

struct es_stream {
  const es_model *model;                  /* read-only, shared, owned by the caller */
  es_fft fft;                             /* this stream's own FFT setup (never shared) */
  es_stream_state st;                     /* zeroed by es_stream_reset */
  /* Scratch, 64-byte aligned for vDSP and BLAS. */
  _Alignas(64) float frame[ES_NFFT];      /* windowed frame, then the inverse FFT output */
  float Xr[ES_NBINS], Xi[ES_NBINS];       /* this hop's spectrum */
  float feat[ES_NFEAT];                   /* features, index 3*bin + f */
  float fwork[3 * ES_NBINS];              /* feature scratch: p, mag, t */
  _Alignas(64) float z[16][64];           /* band activations */
  _Alignas(64) float zn[16][64];          /* their LayerNorm */
  _Alignas(64) float gates[16][512];      /* time-LSTM gates of the 16 bands */
  float tmp[16][128];                     /* tanh(c) scratch of the time-LSTM cell */
  _Alignas(64) float gxf[16][256];        /* band-LSTM input projections, forward */
  _Alignas(64) float gxb[16][256];        /* and reverse */
  float lw[7 * 64];                       /* band-LSTM work: h, c, gates, tanh c (B = 1, H = 64) */
  _Alignas(64) float bcat[16][128];       /* band-LSTM outputs [fwd | bwd] per band */
  _Alignas(64) float xt[128][16];         /* the 16-row products' input, transposed (only the opt-in es_linear_xt path uses it) */
  float q[64], sc[4][ES_MAX_TOKENS], a[64];   /* attention query, scores, output */
  float mu[64], h2[2];                    /* heads: mean over bands and the two logits */
  float hid[16][128];                     /* decoder hidden layer */
  float coef[ES_COEF_TOTAL];              /* deep-filter coefficients after 1.5 tanh */
  float Yr[ES_NBINS], Yi[ES_NBINS];       /* filtered spectrum */
};

/* The real hop (20-engine 6.6). d == NULL is es_stream_hop; with d set, the debug tensors of contract
   3.3 are copied into the non-NULL pointers of d. Returns 1 if the NaN guard reset the stream. */
int es_stream_hop_dbg(es_stream *s, const es_cond *c, const float in[ES_HOP], float out[ES_HOP],
                      float heads[2], const es_dbg *d);
/* The two pieces of the hop that test_units checks against naive double code (20-engine 14.1).
   Cross-attention of 6.4 on z [16][64] in place (q [64], sc [4][68] and a [64] are scratch), and the
   deep filter of 6.5 including the history shift (xhist is the stream's [2][128][2] history). */
void es_attention(const es_model *m, const es_cond *c, float *z, float *q, float *sc, float *a);
void es_deep_filter(const es_model *m, const float *coef, const float *Xr, const float *Xi,
                    float xhist[2][ES_DF_BINS][2], float *Yr, float *Yi);

/* Token encoder (enroll.c, 20-engine 7.2), shared by es_enroll and es_embed_g. mask has
   es_enroll_num_frames(n) entries; limit is 48 (target) or 16 (own). On ES_OK it writes n_tok rows of
   64 into tokens (room for `limit` rows), g [128], and kept = the number of frames with mask != 0.
   ES_ERR_ARG (bad arguments, n > 960,000), ES_ERR_ENROLL_SHORT (fewer than 4 tokens) or ES_ERR_NOMEM. */
es_status es_tokens(const es_model *m, const float *pcm, int n, const uint8_t *mask, int limit,
                    float *tokens, int *n_tok, float g[ES_G_DIM], int *kept);
/* Pooling and selection of 20-engine 7.2 steps 4 and 6 (no network): which 3 frames make up each kept
   token. Writes n_tok rows of 3 frame indices into groups (room for limit rows); returns n_avail
   (before the limit). Exposed for the pooling unit test. */
int es_token_groups(const uint8_t *mask, int F, int limit, int (*groups)[3], int *n_tok, int *kept);

/* Test-vector tolerances (contract 3.3, selftest.c). es_tv_tolerances reads meta.tolerances, which
   must hold exactly the 10 keys below (a missing or extra key fails, never a default), each equal to its
   contract 3.3 value (out / out_snr_db by file length: N = 32,000 short, N = 160,000 long, any other N
   fails). ES_OK or ES_ERR_FORMAT with err naming the key. */
typedef struct { double vad, tokens, g, film, k, v, dbg, out, out_snr_db, heads; } es_tv_tol;
es_status es_tv_tolerances(const es_container *ct, es_tv_tol *tol, char *err, size_t errlen);
/* A malloc'd copy of f32 array `name` when it exists with exactly `count` floats, else NULL. */
float *es_tv_f32(const es_container *ct, const char *name, size_t count);
/* Max abs difference of two float arrays (NaN counts as infinite) and the SNR of 20-engine 13 in dB
   (10 log10(sum ref^2 / sum (x - ref)^2), +inf for a zero error). */
double es_max_abs_diff(const float *x, const float *ref, size_t n);
double es_snr_db(const float *x, const float *ref, size_t n);

/* ------------------------------------------------------------------------------------------------
 * Package E-2: safety-chain helpers (safety.c, 20-engine 11). Render thread only.
 * ---------------------------------------------------------------------------------------------- */
/* es_limiter_process that also returns the smallest gain it used in this block (for limiter_gr_db).
   es_limiter_process calls it, so there is one limiter implementation. */
float es_limiter_process_min(es_limiter *l, float *x, int n);

/* Runaway guard (20-engine 11): output more than 30 dB above the input for 0.2 s ducks by 20 dB. */
typedef struct {
  float sr;              /* I/O rate */
  float in_ms, out_ms;   /* one-pole mean squares (50 ms) of the input and of the output after the trim */
  int64_t over;          /* consecutive I/O samples with the howl condition true */
  int64_t under;         /* consecutive I/O samples with the condition false while howl is set */
  int howl;              /* 1 = guard active (stats.howl) */
  es_ramp duck;          /* 1 normally, 0.1 (-20 dB) while howl */
} es_guard;
void es_guard_init(es_guard *g, float sr);
/* Measure xin (sanitized input) and y (after stage 3), then apply the duck ramp to y in place.
   Returns the howl flag. RT-safe. */
int  es_guard_process(es_guard *g, const float *xin, float *y, int n);

/* Cue beeps (20-engine 11): 1..3 beeps of 60 ms at 880 Hz, -18 dBFS peak, 5 ms raised-cosine edges,
   90 ms of silence between beeps. */
typedef struct {
  float sr, phase, dphase;   /* phase accumulator, wrapped at 2 pi */
  int beep_len, edge_len, period_len;   /* 60 ms, 5 ms and 150 ms (beep + gap) in samples */
  int total;                 /* samples in the running sequence, 0 = idle */
  int pos;                   /* samples of the sequence already played */
} es_cue;
void es_cue_init(es_cue *c, float sr);
void es_cue_start(es_cue *c, int beeps);       /* clamps to 1..3; a new request restarts the sequence */
void es_cue_add(es_cue *c, float *y, int n);   /* adds the next n samples of the sequence to y */

/* ------------------------------------------------------------------------------------------------
 * Package E-2: post-processing per hop at 16 kHz (post.c, 20-engine 10, contract 4.2). Worker only.
 * Every gain moves once per hop; inside the hop it is applied with a per-sample linear ramp from the
 * previous hop's value to the new one. Two gains drop at once instead (no ramp when they fall): the
 * gain cap ("removed at once") and the room envelope ("forced to 0 at once").
 * ---------------------------------------------------------------------------------------------- */
typedef struct {
  float a200, a150, a400, a50;       /* one-pole coefficients per 4 ms hop: 1 - exp(-4 / tau_ms) */
  float g_db[ES_MAX_INSTANCES];      /* (1) activity gate per slot, dB (0 open .. -12 closed) */
  float pw[ES_MAX_INSTANCES];        /* (2) crossfade power per slot, 0..1 */
  float slot_lin[ES_MAX_INSTANCES];  /*     previous hop's sqrt(power) * gate, linear */
  float d_db;                        /* (3) own-voice duck, dB */
  float agc_db, agc_ms;              /* (4) AGC gain (dB) and its 400 ms mean square */
  int   agc_primed;                  /*     0 until the first hop with speech sets agc_ms */
  float main_lin;                    /*     previous hop's duck * AGC, linear */
  float in_ms, out_ms, cap_db;       /* (5) gain cap: 50 ms mean squares of x_d and of (4), cap in dB */
  float cap_lin;                     /*     previous hop's cap, linear */
  float r_env, room_lin;             /* (6) room envelope 0..1 and previous hop's room gain */
  float bm;                          /* (7) bypass mix 0..1 */
  _Alignas(64) float sum[ES_HOP];    /* scratch: the crossfaded sum of (2) */
  float ypre[ES_HOP];                /* scratch: the output of (4), before the cap */
} es_post;

/* Start values: gates open, slot powers at slot_gain^2, no duck, AGC 0 dB, no cap, room envelope 1,
   bypass mix at prm->bypass. */
void es_post_init(es_post *p, const es_params *prm);
/* One hop. ys[s] and hd[s] = {p_target, p_self} are the stream outputs of slot s (zeros for an empty
   slot), active[s] = 1 if slot s has conditioning with n_tokens >= 4, xd = the previous hop's input.
   Writes y[64] and the maxima of p_target and p_self over active slots (0 if none). */
void es_post_hop(es_post *p, const es_params *prm, int n_slots, const float (*ys)[ES_HOP],
                 const float (*hd)[2], const int *active, const float *xd, float *y,
                 float *pt_max, float *ps_max);

/* ------------------------------------------------------------------------------------------------
 * Package E-2: the engine (pipeline.c, rt_thread.c, 20-engine 8). es_engine_new allocates every
 * buffer; after that nothing on the audio thread or the worker allocates, locks or logs.
 * Which thread owns which field is written next to each group.
 * ---------------------------------------------------------------------------------------------- */
#include <pthread.h>
#include <mach/mach.h>
#include "ring.h"

#define ES_IN16_SIZE      4096u    /* contract 4.2 ring sizes */
#define ES_OUT16_SIZE     4096u
#define ES_OUTHIST_SIZE   65536u   /* per-slot output history (reads up to 48,000) */
#define ES_OUTHIST_READ   48000
#define ES_TIMINGS_SIZE   1024u
#define ES_MAX_HOPS_CALL  64       /* most hops one es_engine_run_hops call runs (20-engine 3.8) */
#define ES_SLACK_STEP     21       /* 22 -> 43 -> 64 */
#define ES_PARAM_WORDS    12       /* es_params is 12 four-byte words (seqlock copies them) */

/* Latency probe states (20-engine 12). The render owns RUNNING -> CAPTURED; probe.c owns the rest. */
enum { ES_PROBE_IDLE = 0, ES_PROBE_RUNNING = 1, ES_PROBE_CAPTURED = 2 };

struct es_engine {
  /* ---- Set by es_engine_new, read-only afterwards (any thread) ---- */
  const es_model *model;
  es_engine_config cfg;
  int fam;                           /* resampler family: 0 = 48 kHz, 1 = 44.1 kHz */
  int n_inst;                        /* cfg.n_instances */
  int p_min;                         /* prefill of 20-engine 9.4 for (io_rate, io_frames) */
  uint32_t tb_numer, tb_denom;       /* mach_timebase_info, cached (ticks * numer / denom = ns) */
  uint64_t pol_period, pol_computation, pol_constraint;   /* worker policy in Mach ticks (8.4) */
  int ramp1ms;                       /* I/O samples in 1 ms (rounded up): the conceal ramp length */

  /* ---- Rings (8.1). Each has one producer and one consumer, see the table in 20-engine 8.1 ---- */
  es_ring in16, out16;               /* render -> worker, worker -> render */
  es_hist inhist;                    /* render -> es_engine_read_input */
  es_hist outhist[ES_MAX_INSTANCES]; /* worker -> es_engine_read_output (raw stream output per slot) */
  es_ring16 timings;                 /* worker -> es_engine_read_timings */
  float *in16_buf, *out16_buf, *inhist_buf, *outhist_buf[ES_MAX_INSTANCES];
  uint16_t *timings_buf;

  /* ---- Worker thread (rt_thread.c) ---- */
  semaphore_t sem;                   /* render signals it when in16 holds a hop */
  int sem_made;
  pthread_t thread;
  _Atomic int worker_running;        /* 1 while the worker thread loops */
  _Atomic int worker_stop;           /* set by es_engine_stop_worker */
  _Atomic int manual_worker;         /* test hook: the caller runs es_engine_run_hops itself */
  _Atomic uint64_t worker_wakes;     /* worker loop wake-ups (relaxed; test_rt compares them with the
                                        worker's semaphore waits, so a wait inside run_hops shows) */
  semaphore_t start_sem;             /* the new thread signals it after its setup */
  es_status start_status;            /* setup result written by the new thread before start_sem */
  void *workgroup;                   /* os_workgroup_t to join, or NULL */
  int wg_join_result;                /* 0 = joined (or no workgroup), else the os_workgroup_join error */

  /* ---- Conditioning hand-off (contract 4.3, 20-engine 8.5) ---- */
  es_cond *cond[ES_MAX_INSTANCES][2];          /* written by set_cond only while not active */
  _Atomic int active[ES_MAX_INSTANCES];        /* set_cond stores (release), worker loads (acquire) */
  _Atomic int ack[ES_MAX_INSTANCES];           /* worker stores (release), set_cond loads (acquire) */
  _Atomic int reset_pending[ES_MAX_INSTANCES][2];   /* per buffer: reset the stream before using it */
  int last_a[ES_MAX_INSTANCES];                /* worker: buffer index used by the previous hop */
  uint64_t n_switches[ES_MAX_INSTANCES];       /* worker: buffer changes seen (hand-off test) */
  uint64_t n_resets[ES_MAX_INSTANCES];         /* worker: resets applied (hand-off test) */

  /* ---- Parameters (seqlock, 8.5), mute and cue requests (one atomic int each) ---- */
  _Atomic uint32_t pseq;                       /* odd while set_params is writing */
  _Atomic uint32_t pwords[ES_PARAM_WORDS];     /* the published es_params, word by word */
  _Atomic int mute_req;                        /* 1 = muted (es_engine_set_mute) */
  _Atomic int cue_req;                         /* 0 = none, else beeps requested (es_engine_cue) */

  /* ---- Worker-only state ---- */
  es_stream *st[ES_MAX_INSTANCES];
  es_params wp;                                /* the worker's copy of the params */
  es_post post;
  _Alignas(64) float x[ES_HOP];                /* this hop's input */
  float xd[ES_HOP];                            /* the previous hop's input (8.6) */
  float dl[ES_HOP];                            /* delay line that makes xd */
  float ys[ES_MAX_INSTANCES][ES_HOP];          /* stream outputs */
  float y[ES_HOP];                             /* post output */
  float hd[ES_MAX_INSTANCES][2];               /* heads {p_target, p_self} per slot */
  int act[ES_MAX_INSTANCES];                   /* slot ran this hop (n_tokens >= 4) */

  /* ---- Render-only state (audio thread) ---- */
  es_params rp;                                /* the render's copy of the params */
  es_rs dec, itp;                              /* decimator (I/O -> 16 kHz), interpolator (16 kHz -> I/O) */
  float *dec_hist, *itp_hist;
  float *xin, *d16, *u16, *y48, *g48;          /* scratch: 4,096 / 1,600 / 1,600 / 4,096 / 4,096 */
  int64_t S;                                   /* 16 kHz samples the interpolator consumed (real or zero) */
  int64_t ins, din;                            /* applied planned zeros: slack steps, dropped input */
  int64_t ins_pending, din_pending;            /* planned zeros waiting for the conceal ramp to reach 0 */
  int grow_pending;                            /* a slack step waits for all p_target < 0.2 */
  int slack16;                                 /* current slack (render copy; stats hold the public one) */
  uint64_t io_pos;                             /* I/O samples rendered so far (underrun timestamps) */
  int64_t last_underrun;                       /* io_pos of the previous counted underrun, -1 = none */
  int cb_underrun;                             /* 1 once this callback counted its underrun */
  int mute_applied;                            /* mute state the mute ramp was last armed for */
  float trim_target;                           /* linear out_gain the trim ramp heads to */
  es_ramp conceal, mute, trim;
  es_limiter lim;
  es_guard guard;
  es_cue cue;

  /* ---- Stats: every field atomic, relaxed (8.7); render writes some, the worker the rest ---- */
  _Atomic uint64_t s_hops, s_underruns, s_nan_resets, s_probe_runs;
  _Atomic int32_t s_slack16, s_last_frames, s_howl;
  _Atomic float s_p_target[ES_MAX_INSTANCES], s_p_self;
  _Atomic float s_in_rms_db, s_out_rms_db, s_agc_gain_db, s_limiter_gr_db;
  _Atomic float s_compute_us_last, s_compute_us_max;

  /* ---- Latency probe (20-engine 12). pipeline.c builds the sweep, plays it and captures the mic
     (render step 8); probe.c (es_engine_probe_start / _result) starts runs and analyzes them. ---- */
  _Atomic int probe_state;                     /* ES_PROBE_* (store with release, load with acquire) */
  float *probe_sweep;                          /* one sweep at io_rate, probe_sweep_len samples (0.3 s) */
  int probe_sweep_len;
  int probe_period;                            /* samples from one sweep start to the next (0.8 s) */
  int probe_play_len;                          /* playback length (4.0 s): 5 sweeps */
  float *probe_cap;                            /* mic capture at io_rate, probe_cap_len samples (4.5 s) */
  int probe_cap_len;
  int probe_pos;                               /* samples played and captured; render-only while RUNNING */
  void *probe_priv;                            /* free for probe.c (a cached result etc.): NULL at create,
                                                  one malloc'd block, es_engine_free calls free() on it */
};

/* Test and tool hook (tests, esrun, fuzz): on = 1 says the caller runs es_engine_run_hops itself
   (no worker thread). Underruns then count, and set_cond keeps the BUSY rule as if a worker ran. */
void es_engine_set_manual_worker(es_engine *e, int on);
/* Mach ticks to microseconds with the engine's cached timebase. */
uint64_t es_ticks_to_us(const es_engine *e, uint64_t ticks);
/* Probe analysis with a chosen GCC-PHAT band (probe.c; non-RT, allocates). The engine uses 300 Hz to
   6 kHz (DECISIONS L 2026-11-08); test_units also runs the full band to compare. Fills the lag and
   psr of each of the 5 sweeps from the last capture. ES_OK or ES_ERR_NOMEM. */
es_status es_probe_analyze_band(const es_engine *e, double lo_hz, double hi_hz, float lag_ms[5], float psr_db[5]);

#endif /* EARSHOT_INTERNAL_H */
