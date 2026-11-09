/* model.c - the .esw / .est container reader and the canonical model config (20-engine 4.1, 4.2,
 * contract 3.1).
 *
 * Package E-0 wrote the container reader (es_container_open/find/close), the compiled config check,
 * the SHA-256 helper, the list of the 270 weight tensors (es_model_tensors, contract 3.2), the
 * synthetic test models (es_model_synthetic, 20-engine 4.4), es_model_free and the getters.
 * Package E-1 added the tensor binding (bind), es_model_load from memory and es_model_load_file
 * (20-engine 4.3) at the end of this file; bind walks the same tensor list as the synthetic model.
 * Threads: any non-RT thread (the reader and the model builders allocate).
 *
 * File layout (all little-endian):
 *   offset 0   4 bytes  magic "ESW1" (weights) or "EST1" (test vectors)
 *   offset 4   uint32   format_version = 1
 *   offset 8   uint32   header_len (bytes of JSON that follow, no NUL)
 *   offset 12  uint32   reserved = 0
 *   offset 16  JSON header
 *   data_start = round_up(16 + header_len, 64): the arrays, each at a 64-byte aligned offset
 */
#include <CommonCrypto/CommonDigest.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

/* The config exactly as Python writes it with json.dumps(config, sort_keys=True, separators=(",", ":"))
   (20-engine 4.2). 470 bytes; its SHA-256 is ES_CONFIG_HASH_HEX (internal.h), checked by test_units. */
const char ES_CONFIG_CANON[] =
  "{\"D\":64,\"ana_rise\":448,\"att_after\":0,\"att_dim\":64,\"att_heads\":4,\"band_hidden\":64,"
  "\"band_widths\":[8,8,8,8,8,8,8,8,16,16,16,16,32,32,32,33],\"blocks\":4,\"df_bands\":12,\"df_taps\":3,"
  "\"feat_log_scale\":0.1,\"feat_pow\":0.3,\"film_after\":[0,2],\"g_dim\":128,\"hop\":64,\"ln_eps\":1e-05,"
  "\"mask_bound\":1.5,\"max_own_tokens\":16,\"max_target_tokens\":48,\"n_fft\":512,\"null_tokens\":4,"
  "\"sr\":16000,\"synth_len\":128,\"time_hidden\":128,\"tok_dim\":64,\"tok_hidden\":32,\"tok_hop\":512,"
  "\"tok_pool\":3,\"wespk_dim\":256}";

/* The same config as compiled constants, one row per key (contract 3.1: exactly these 29 keys).
   kind 0 = integer (must match exactly), 1 = float (relative 1e-12), 2 = integer array. */
typedef struct { const char *key; int kind; double value; const int *arr; int arr_len; } cfg_entry;

static const int CFG_BAND_WIDTHS[16] = { 8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 33 };
static const int CFG_FILM_AFTER[2] = { 0, 2 };

static const cfg_entry CFG[29] = {
  { "D", 0, 64, NULL, 0 },               { "ana_rise", 0, 448, NULL, 0 },
  { "att_after", 0, 0, NULL, 0 },        { "att_dim", 0, 64, NULL, 0 },
  { "att_heads", 0, 4, NULL, 0 },        { "band_hidden", 0, 64, NULL, 0 },
  { "band_widths", 2, 0, CFG_BAND_WIDTHS, 16 },
  { "blocks", 0, 4, NULL, 0 },           { "df_bands", 0, 12, NULL, 0 },
  { "df_taps", 0, 3, NULL, 0 },          { "feat_log_scale", 1, 0.1, NULL, 0 },
  { "feat_pow", 1, 0.3, NULL, 0 },       { "film_after", 2, 0, CFG_FILM_AFTER, 2 },
  { "g_dim", 0, 128, NULL, 0 },          { "hop", 0, 64, NULL, 0 },
  { "ln_eps", 1, 1e-05, NULL, 0 },       { "mask_bound", 1, 1.5, NULL, 0 },
  { "max_own_tokens", 0, 16, NULL, 0 },  { "max_target_tokens", 0, 48, NULL, 0 },
  { "n_fft", 0, 512, NULL, 0 },          { "null_tokens", 0, 4, NULL, 0 },
  { "sr", 0, 16000, NULL, 0 },           { "synth_len", 0, 128, NULL, 0 },
  { "time_hidden", 0, 128, NULL, 0 },    { "tok_dim", 0, 64, NULL, 0 },
  { "tok_hidden", 0, 32, NULL, 0 },      { "tok_hop", 0, 512, NULL, 0 },
  { "tok_pool", 0, 3, NULL, 0 },         { "wespk_dim", 0, 256, NULL, 0 },
};

/* SHA-256 of n bytes as 64 lowercase hex characters plus a NUL. Fed in 1 GB pieces because
   CommonCrypto takes 32-bit lengths. */
void es_sha256_hex(const void *data, size_t n, char out[65]) {
  CC_SHA256_CTX ctx;
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  const unsigned char *p = data;
  CC_SHA256_Init(&ctx);
  while (n > 0) {
    size_t piece = n > (1u << 30) ? (1u << 30) : n;
    CC_SHA256_Update(&ctx, p, (CC_LONG)piece);
    p += piece;
    n -= piece;
  }
  CC_SHA256_Final(digest, &ctx);
  static const char hex[] = "0123456789abcdef";
  for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) {
    out[2 * i] = hex[digest[i] >> 4];
    out[2 * i + 1] = hex[digest[i] & 15];
  }
  out[64] = '\0';
}

/* Little-endian uint32 read with memcpy (never a pointer cast, the buffer may be unaligned). */
static uint32_t read_u32(const uint8_t *p) {
  uint8_t b[4];
  memcpy(b, p, 4);
  return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

/* Write one line into err (if given) and return the status. */
static es_status fail(char *err, size_t errlen, es_status st, const char *fmt, ...) {
  if (err && errlen > 0) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
  }
  return st;
}

/* Read a JSON number that must be a whole number in 0 .. 2^53. Returns 0 or -1. */
static int json_uint(const es_json *j, int t, uint64_t *out) {
  double v;
  if (es_json_num(j, t, &v) != 0) return -1;
  if (v < 0.0 || v > 9007199254740992.0 || floor(v) != v) return -1;
  *out = (uint64_t)v;
  return 0;
}

/* Copy string token t into a C buffer for error messages (cut to fit). */
static void tok_text(const es_json *j, int t, char *buf, size_t cap) {
  int n = (t >= 0 && t < j->n_tok) ? j->tok[t].end - j->tok[t].start : 0;
  if (n < 0) n = 0;
  if ((size_t)n >= cap) n = (int)cap - 1;
  if (n > 0) memcpy(buf, j->js + j->tok[t].start, (size_t)n);
  buf[n] = '\0';
}

/* Check 6 of 20-engine 4.1: the header's config object has exactly the 29 keys of contract 3.1 and each
   equals the compiled constant. The error names the first bad field, e.g. "config.hop: 32, expected 64". */
static es_status check_config(const es_json *j, int cfg, char *err, size_t errlen) {
  char text[80];
  if (cfg < 0 || j->tok[cfg].type != ES_JSON_OBJECT)
    return fail(err, errlen, ES_ERR_CONFIG, "config: missing or not an object");

  /* Every key in the file must be one we know. */
  int key = cfg + 1;
  for (int c = 0; c < j->tok[cfg].size; c++) {
    int known = 0;
    for (int e = 0; e < 29; e++)
      if (es_json_str_eq(j, key, CFG[e].key)) known = 1;
    if (!known) {
      tok_text(j, key, text, sizeof(text));
      return fail(err, errlen, ES_ERR_CONFIG, "config.%s: unexpected key", text);
    }
    key = j->tok[key].next;
  }

  /* Every key we know must be present with the compiled value. */
  for (int e = 0; e < 29; e++) {
    const cfg_entry *ce = &CFG[e];
    int v = es_json_get(j, cfg, ce->key);
    if (v < 0) return fail(err, errlen, ES_ERR_CONFIG, "config.%s: missing", ce->key);
    tok_text(j, v, text, sizeof(text));
    if (ce->kind == 2) {                                   /* integer array, element by element */
      if (j->tok[v].type != ES_JSON_ARRAY || j->tok[v].size != ce->arr_len)
        return fail(err, errlen, ES_ERR_CONFIG, "config.%s: %s, expected %d integers", ce->key, text, ce->arr_len);
      for (int i = 0; i < ce->arr_len; i++) {
        double x;
        if (es_json_num(j, es_json_arr_at(j, v, i), &x) != 0 || x != (double)ce->arr[i])
          return fail(err, errlen, ES_ERR_CONFIG, "config.%s[%d]: expected %d", ce->key, i, ce->arr[i]);
      }
    } else {
      double x;
      if (es_json_num(j, v, &x) != 0)
        return fail(err, errlen, ES_ERR_CONFIG, "config.%s: %s is not a number", ce->key, text);
      int ok = ce->kind == 0 ? (x == ce->value) : (fabs(x - ce->value) <= 1e-12 * fabs(ce->value));
      if (!ok) return fail(err, errlen, ES_ERR_CONFIG, "config.%s: %s, expected %.17g", ce->key, text, ce->value);
    }
  }

  /* Exactly 29 keys (catches a key written twice). */
  if (j->tok[cfg].size != 29)
    return fail(err, errlen, ES_ERR_CONFIG, "config: %d keys, expected 29", j->tok[cfg].size);
  return ES_OK;
}

/* Check 5: "1.x.y" where x and y are digits. Returns 1 if the major version is 1. */
static int model_version_ok(const es_json *j, int t) {
  int n = es_json_str_len(j, t);
  if (n < 5) return 0;
  const char *s = j->js + j->tok[t].start;
  if (s[0] != '1' || s[1] != '.') return 0;
  int i = 2, dots = 0, digits = 0;
  for (; i < n; i++) {
    if (s[i] == '.') {
      if (digits == 0) return 0;
      dots++;
      digits = 0;
    } else if (s[i] >= '0' && s[i] <= '9') {
      digits++;
    } else {
      return 0;
    }
  }
  return dots == 1 && digits > 0;
}

/* Check 10 for one entry of the "arrays" list. */
static es_status read_array(es_container *ct, int i, int e, size_t *name_pos, uint64_t *prev_end,
                            char *err, size_t errlen) {
  const es_json *j = &ct->j;
  es_array *a = &ct->arrays[i];
  if (e < 0 || j->tok[e].type != ES_JSON_OBJECT)
    return fail(err, errlen, ES_ERR_FORMAT, "arrays[%d]: not an object", i);

  /* name: 1..255 plain bytes, copied with a NUL into ct->names */
  int nt = es_json_get(j, e, "name");
  int nlen = es_json_str_len(j, nt);
  if (nlen <= 0 || nlen > 255 || memchr(j->js + j->tok[nt].start, '\\', (size_t)nlen))
    return fail(err, errlen, ES_ERR_FORMAT, "arrays[%d].name: missing or not a plain string", i);
  char *name = ct->names + *name_pos;
  memcpy(name, j->js + j->tok[nt].start, (size_t)nlen);
  name[nlen] = '\0';
  *name_pos += (size_t)nlen + 1;
  a->name = name;
  for (int k = 0; k < i; k++)
    if (strcmp(ct->arrays[k].name, name) == 0)
      return fail(err, errlen, ES_ERR_FORMAT, "array %s: duplicate name", name);

  /* dtype */
  int dt = es_json_get(j, e, "dtype");
  size_t item;
  if (es_json_str_eq(j, dt, "f32"))      { a->dtype = ES_DT_F32; item = 4; }
  else if (es_json_str_eq(j, dt, "u8"))  { a->dtype = ES_DT_U8;  item = 1; }
  else if (es_json_str_eq(j, dt, "i32")) { a->dtype = ES_DT_I32; item = 4; }
  else return fail(err, errlen, ES_ERR_FORMAT, "array %s: dtype is not f32, u8 or i32", name);

  /* shape: up to 4 whole numbers; count = their product */
  int sh = es_json_get(j, e, "shape");
  if (sh < 0 || j->tok[sh].type != ES_JSON_ARRAY || j->tok[sh].size > 4)
    return fail(err, errlen, ES_ERR_FORMAT, "array %s: shape missing or more than 4 dims", name);
  a->ndim = j->tok[sh].size;
  uint64_t count = 1;
  for (int d = 0; d < a->ndim; d++) {
    uint64_t dim;
    if (json_uint(j, es_json_arr_at(j, sh, d), &dim) != 0 || dim > 0x7fffffff)
      return fail(err, errlen, ES_ERR_FORMAT, "array %s: shape[%d] is not a valid size", name, d);
    a->shape[d] = (int)dim;
    if (dim != 0 && count > (UINT64_C(1) << 52) / dim)
      return fail(err, errlen, ES_ERR_FORMAT, "array %s: shape too large", name);
    count *= dim;
  }

  /* offset and nbytes */
  uint64_t off, nb;
  if (json_uint(j, es_json_get(j, e, "offset"), &off) != 0)
    return fail(err, errlen, ES_ERR_FORMAT, "array %s: offset missing", name);
  if (json_uint(j, es_json_get(j, e, "nbytes"), &nb) != 0)
    return fail(err, errlen, ES_ERR_FORMAT, "array %s: nbytes missing", name);
  if (off % 64 != 0)
    return fail(err, errlen, ES_ERR_FORMAT, "array %s: offset %llu is not a multiple of 64",
                name, (unsigned long long)off);
  if (nb != count * item)
    return fail(err, errlen, ES_ERR_FORMAT, "array %s: nbytes %llu, expected %llu", name,
                (unsigned long long)nb, (unsigned long long)(count * item));
  if (off > ct->data_bytes || nb > ct->data_bytes - off)
    return fail(err, errlen, ES_ERR_FORMAT, "array %s: offset %llu + nbytes %llu past the %zu data bytes",
                name, (unsigned long long)off, (unsigned long long)nb, ct->data_bytes);
  if (off < *prev_end)
    return fail(err, errlen, ES_ERR_FORMAT, "array %s: offset %llu before the end of the previous array",
                name, (unsigned long long)off);
  *prev_end = off + nb;
  a->offset = (size_t)off;
  a->nbytes = (size_t)nb;
  a->data = ct->base + ct->data_start + off;
  return ES_OK;
}

/* All checks of 20-engine 4.1 in order. On failure the caller closes the container. */
static es_status open_checks(const uint8_t *b, size_t n, const char magic[4], es_container *ct,
                             char *err, size_t errlen) {
  int weights = memcmp(magic, "ESW1", 4) == 0;

  /* 1. Size and magic. */
  if (n < 16) return fail(err, errlen, ES_ERR_FORMAT, "file: %zu bytes, shorter than the 16-byte preamble", n);
  if (memcmp(b, magic, 4) != 0) return fail(err, errlen, ES_ERR_FORMAT, "magic: expected %.4s", magic);

  /* 2. Container format version. */
  uint32_t version = read_u32(b + 4);
  if (version != 1) return fail(err, errlen, ES_ERR_VERSION, "format_version: %u, expected 1", version);

  /* 3. Header bounds, reserved word, one JSON object. */
  uint32_t header_len = read_u32(b + 8), reserved = read_u32(b + 12);
  if ((uint64_t)16 + header_len > (uint64_t)n)
    return fail(err, errlen, ES_ERR_FORMAT, "header_len: %u bytes run past the end of the %zu-byte file",
                header_len, n);
  if (reserved != 0) return fail(err, errlen, ES_ERR_FORMAT, "reserved: %u, expected 0", reserved);
  es_status st = es_json_parse(&ct->j, (const char *)b + 16, header_len);
  if (st == ES_ERR_NOMEM) return fail(err, errlen, st, "header: out of memory");
  if (st != ES_OK) return fail(err, errlen, ES_ERR_FORMAT, "header: not valid JSON");
  const es_json *j = &ct->j;
  if (j->tok[0].type != ES_JSON_OBJECT) return fail(err, errlen, ES_ERR_FORMAT, "header: not a JSON object");

  /* 4. format, format_version, model. */
  const char *want_format = weights ? "earshot-weights" : "earshot-testvec";
  if (!es_json_str_eq(j, es_json_get(j, 0, "format"), want_format))
    return fail(err, errlen, ES_ERR_FORMAT, "format: expected %s", want_format);
  uint64_t fv;
  if (json_uint(j, es_json_get(j, 0, "format_version"), &fv) != 0 || fv != 1)
    return fail(err, errlen, ES_ERR_FORMAT, "header format_version: expected 1");
  if (!es_json_str_eq(j, es_json_get(j, 0, "model"), "esb16-df"))
    return fail(err, errlen, ES_ERR_FORMAT, "model: expected esb16-df");

  /* 5. model_version major 1. */
  if (!model_version_ok(j, es_json_get(j, 0, "model_version")))
    return fail(err, errlen, ES_ERR_VERSION, "model_version: expected 1.x.y");

  /* 6. config field by field. 7. config_hash. */
  st = check_config(j, es_json_get(j, 0, "config"), err, errlen);
  if (st != ES_OK) return st;
  es_sha256_hex(ES_CONFIG_CANON, strlen(ES_CONFIG_CANON), ct->config_hash);
  if (!es_json_str_eq(j, es_json_get(j, 0, "config_hash"), ct->config_hash))
    return fail(err, errlen, ES_ERR_CONFIG, "config_hash: does not match the engine's config");

  /* 8. data_start and data_bytes. */
  ct->data_start = ((size_t)16 + header_len + 63) / 64 * 64;
  uint64_t data_bytes;
  if (json_uint(j, es_json_get(j, 0, "data_bytes"), &data_bytes) != 0)
    return fail(err, errlen, ES_ERR_FORMAT, "data_bytes: missing");
  if (ct->data_start > n || data_bytes != (uint64_t)(n - ct->data_start))
    return fail(err, errlen, ES_ERR_FORMAT, "data_bytes: %llu, but the file holds %zu bytes after data_start",
                (unsigned long long)data_bytes, ct->data_start > n ? (size_t)0 : n - ct->data_start);
  ct->data_bytes = (size_t)data_bytes;

  /* 9. data_sha256. */
  es_sha256_hex(b + ct->data_start, ct->data_bytes, ct->data_hash);
  if (!es_json_str_eq(j, es_json_get(j, 0, "data_sha256"), ct->data_hash))
    return fail(err, errlen, ES_ERR_FORMAT, "data_sha256: does not match the data");

  /* 10. The arrays list. */
  int arr = es_json_get(j, 0, "arrays");
  if (arr < 0 || j->tok[arr].type != ES_JSON_ARRAY)
    return fail(err, errlen, ES_ERR_FORMAT, "arrays: missing or not a list");
  int na = j->tok[arr].size;
  ct->arrays = calloc((size_t)(na > 0 ? na : 1), sizeof(es_array));
  ct->names = malloc((size_t)header_len + 1);       /* all names together are shorter than the header */
  if (!ct->arrays || !ct->names) return fail(err, errlen, ES_ERR_NOMEM, "arrays: out of memory");
  size_t name_pos = 0;
  uint64_t prev_end = 0;
  int e = arr + 1;                                   /* walk the list element by element */
  for (int i = 0; i < na; i++) {
    st = read_array(ct, i, e, &name_pos, &prev_end, err, errlen);
    if (st != ES_OK) return st;
    ct->n_arrays = i + 1;
    e = j->tok[e].next;
  }
  return ES_OK;
}

es_status es_container_open(const void *buf, size_t n, const char magic[4], es_container *ct,
                            char *err, size_t errlen) {
  if (err && errlen > 0) err[0] = '\0';
  if (!ct) return fail(err, errlen, ES_ERR_ARG, "argument: container is NULL");
  memset(ct, 0, sizeof(*ct));
  if (!buf || !magic) return fail(err, errlen, ES_ERR_ARG, "argument: NULL buffer or magic");
  if (memcmp(magic, "ESW1", 4) != 0 && memcmp(magic, "EST1", 4) != 0)
    return fail(err, errlen, ES_ERR_ARG, "argument: magic must be ESW1 or EST1");
  ct->base = buf;
  ct->n = n;
  es_status st = open_checks(buf, n, magic, ct, err, errlen);
  if (st != ES_OK) es_container_close(ct);
  return st;
}

const es_array *es_container_find(const es_container *ct, const char *name) {
  if (!ct || !name) return NULL;
  for (int i = 0; i < ct->n_arrays; i++)
    if (strcmp(ct->arrays[i].name, name) == 0) return &ct->arrays[i];
  return NULL;
}

void es_container_close(es_container *ct) {
  if (!ct) return;
  es_json_free(&ct->j);
  free(ct->arrays);
  free(ct->names);
  ct->arrays = NULL;
  ct->names = NULL;
  ct->n_arrays = 0;
}

/* ------------------------------------------------------------------------------------------------
 * The 270 weight tensors (contract 3.2). One list, built in the file order of the exporter, serves
 * both the synthetic model below and es_model_load (E-1), so the two can never disagree on a name,
 * a shape or which es_model field a tensor feeds.
 * ---------------------------------------------------------------------------------------------- */
typedef struct { es_tensor *list; int n; } tensor_list;

/* Append one tensor named prefix + suffix. */
static void add(tensor_list *L, const char *prefix, const char *suffix, int d0, int d1, int init,
                const float **dst) {
  if (L->n >= ES_N_TENSORS) return;              /* cannot happen: the list below has exactly 270 */
  es_tensor *t = &L->list[L->n++];
  snprintf(t->name, sizeof(t->name), "%s%s", prefix, suffix);
  t->d0 = d0;
  t->d1 = d1;
  t->init = init;
  t->dst = dst;
}

/* A Linear layer: prefix.weight [in, out] and prefix.bias [out]. */
static void add_lin(tensor_list *L, const char *prefix, int in, int out, es_lin *lin) {
  add(L, prefix, ".weight", in, out, ES_INIT_WEIGHT, &lin->w);
  add(L, prefix, ".bias", out, 0, ES_INIT_BIAS, &lin->b);
}

/* A LayerNorm over 64 channels: prefix.weight [64] and prefix.bias [64]. */
static void add_ln(tensor_list *L, const char *prefix, es_ln *ln) {
  add(L, prefix, ".weight", 64, 0, ES_INIT_LN_WEIGHT, &ln->w);
  add(L, prefix, ".bias", 64, 0, ES_INIT_BIAS, &ln->b);
}

/* One LSTM direction with input size in and hidden size H: prefix.w_ih [in, 4H], prefix.w_hh [H, 4H],
   prefix.bias [4H]; the reverse direction (rev = 1) adds the suffix _rev to each name. */
static void add_lstm(tensor_list *L, const char *prefix, int rev, int in, int H, es_lstmw *w) {
  add(L, prefix, rev ? ".w_ih_rev" : ".w_ih", in, 4 * H, ES_INIT_WEIGHT, &w->wih);
  add(L, prefix, rev ? ".w_hh_rev" : ".w_hh", H, 4 * H, ES_INIT_WEIGHT, &w->whh);
  add(L, prefix, rev ? ".bias_rev" : ".bias", 4 * H, 0, ES_INIT_BIAS, &w->bias);
}

int es_model_tensors(es_model *m, es_tensor *list) {
  tensor_list L = { list, 0 };
  char p[32];
  for (int k = 0; k < 16; k++) {                 /* band encoder: 4 tensors per band */
    snprintf(p, sizeof(p), "enc.lin.%d", k);
    add_lin(&L, p, 3 * CFG_BAND_WIDTHS[k], 64, &m->enc_lin[k]);
    snprintf(p, sizeof(p), "enc.ln.%d", k);
    add_ln(&L, p, &m->enc_ln[k]);
  }
  for (int b = 0; b < 4; b++) {                  /* blocks: 17 tensors each */
    snprintf(p, sizeof(p), "blocks.%d.tn", b);
    add_ln(&L, p, &m->blk[b].tn);
    snprintf(p, sizeof(p), "blocks.%d.t", b);
    add_lstm(&L, p, 0, 64, 128, &m->blk[b].t);   /* time LSTM, hidden 128 */
    snprintf(p, sizeof(p), "blocks.%d.tf", b);
    add_lin(&L, p, 128, 64, &m->blk[b].tf);
    snprintf(p, sizeof(p), "blocks.%d.bn", b);
    add_ln(&L, p, &m->blk[b].bn);
    snprintf(p, sizeof(p), "blocks.%d.b", b);
    add_lstm(&L, p, 0, 64, 64, &m->blk[b].b);    /* band BiLSTM, hidden 64 each way */
    add_lstm(&L, p, 1, 64, 64, &m->blk[b].b_rev);
    snprintf(p, sizeof(p), "blocks.%d.bf", b);
    add_lin(&L, p, 128, 64, &m->blk[b].bf);
  }
  add_lin(&L, "att.q", 1024, 64, &m->att_q);     /* cross-attention */
  add_lin(&L, "att.o", 64, 1024, &m->att_o);
  add_lin(&L, "att.kv", 64, 128, &m->att_kv);
  add(&L, "att.type_emb", "", 3, 64, ES_INIT_EMBED, &m->type_emb);
  add(&L, "att.null_tok", "", 4, 64, ES_INIT_EMBED, &m->null_tok);
  add_lin(&L, "film.0", 770, 256, &m->film0);    /* FiLM generator */
  add_lin(&L, "film.2", 256, 4096, &m->film2);
  add(&L, "null.e_t", "", 256, 0, ES_INIT_EMBED, &m->null_e_t);
  add(&L, "null.g_t", "", 128, 0, ES_INIT_EMBED, &m->null_g_t);
  add(&L, "null.e_o", "", 256, 0, ES_INIT_EMBED, &m->null_e_o);
  add(&L, "null.g_o", "", 128, 0, ES_INIT_EMBED, &m->null_g_o);
  for (int k = 0; k < 16; k++) {                 /* band decoders: 6 tensors per band */
    int n_coef = 2 * CFG_BAND_WIDTHS[k] * (k < 12 ? 3 : 1);   /* 2 W_k T_k */
    snprintf(p, sizeof(p), "dec.%d.ln", k);
    add_ln(&L, p, &m->dec[k].ln);
    snprintf(p, sizeof(p), "dec.%d.fc1", k);
    add_lin(&L, p, 64, 128, &m->dec[k].fc1);
    snprintf(p, sizeof(p), "dec.%d.fc2", k);
    add_lin(&L, p, 128, n_coef, &m->dec[k].fc2);
  }
  add_lin(&L, "heads", 64, 2, &m->heads);
  add_ln(&L, "tok.blk.tn", &m->tok.tn);          /* token encoder (enrollment) */
  add_ln(&L, "tok.blk.bn", &m->tok.bn);
  add_lstm(&L, "tok.blk.t", 0, 64, 32, &m->tok.t);
  add_lstm(&L, "tok.blk.t", 1, 64, 32, &m->tok.t_rev);
  add_lstm(&L, "tok.blk.b", 0, 64, 32, &m->tok.b);
  add_lstm(&L, "tok.blk.b", 1, 64, 32, &m->tok.b_rev);
  add_lin(&L, "tok.blk.tf", 64, 64, &m->tok.tf);
  add_lin(&L, "tok.blk.bf", 64, 64, &m->tok.bf);
  add_lin(&L, "tok.proj", 1024, 64, &m->tok_proj);
  add_lin(&L, "tok.g", 64, 128, &m->tok_g);
  return L.n;
}

/* ------------------------------------------------------------------------------------------------
 * Model objects: fixed tables, the synthetic test model (20-engine 4.4), free and getters.
 * ---------------------------------------------------------------------------------------------- */

/* What every model holds besides its weights (20-engine 4.3): the band layout and the windows.
   es_model_load (E-1) calls this too. */
static void model_fixed_tables(es_model *m) {
  int start = 0, off = 0;
  for (int k = 0; k < 16; k++) {
    m->band_start[k] = start;                    /* 0, 8, ..., 224 */
    m->band_width[k] = CFG_BAND_WIDTHS[k];
    m->band_taps[k] = k < 12 ? 3 : 1;            /* the first df_bands = 12 bands use 3 taps */
    m->coef_off[k] = off;                        /* prefix sums of 2 W_k T_k, total 1,026 */
    start += m->band_width[k];
    off += 2 * m->band_width[k] * m->band_taps[k];
  }
  es_make_windows(m->wa, m->ws);
}

/* Byte offset of every tensor in the arena, the exporter's layout (contract 3.1): in list order, each
   starting at the next multiple of 64 bytes. Returns the bytes the arena needs. */
static size_t tensor_offsets(const es_tensor *list, int n, size_t *off) {
  size_t pos = 0;
  for (int i = 0; i < n; i++) {
    pos = (pos + 63) / 64 * 64;
    off[i] = pos;
    pos += sizeof(float) * (size_t)list[i].d0 * (size_t)(list[i].d1 > 0 ? list[i].d1 : 1);
  }
  return pos;
}

/* xorshift32 (20-engine 4.4), then the top 24 bits mapped to a uniform float in [-1, 1).
   (x >> 8) is below 2^24, so the float is exact. */
static float synth_uniform(uint32_t *x) {
  *x ^= *x << 13;
  *x ^= *x >> 17;
  *x ^= *x << 5;
  return (float)(*x >> 8) * (1.0f / 8388608.0f) - 1.0f;     /* (x >> 8) / 2^23 - 1 */
}

/* Values are drawn in tensor-list order from one xorshift32 stream, so a seed always gives the same
   model. Seed 0 would stay 0 forever in xorshift32, so it is replaced by 2463534242 (Marsaglia's
   example seed). The all-pass version draws the same stream and then overwrites the decoder output
   layers, so every other tensor is identical to the allpass = 0 model of the same seed. */
es_status es_model_synthetic(uint32_t seed, int allpass, es_model **out) {
  if (!out) return ES_ERR_ARG;
  *out = NULL;
  es_model *m = calloc(1, sizeof(es_model));
  es_tensor *list = malloc(sizeof(es_tensor) * ES_N_TENSORS);
  size_t *off = malloc(sizeof(size_t) * ES_N_TENSORS);
  void *arena = NULL;
  int n = 0;
  if (m && list && off) {
    n = es_model_tensors(m, list);
    m->arena_bytes = tensor_offsets(list, n, off);
    if (posix_memalign(&arena, 64, m->arena_bytes) != 0) arena = NULL;
  }
  if (!arena) {                                    /* out of memory somewhere above */
    free(m); free(list); free(off);
    return ES_ERR_NOMEM;
  }
  m->arena = arena;
  memset(m->arena, 0, m->arena_bytes);             /* the gaps between tensors are zero, as in a file */
  model_fixed_tables(m);

  /* Fill every tensor and point its es_model field at it. */
  uint32_t x = seed != 0 ? seed : 2463534242u;
  for (int i = 0; i < n; i++) {
    const es_tensor *t = &list[i];
    float *w = (float *)((char *)m->arena + off[i]);
    int count = t->d0 * (t->d1 > 0 ? t->d1 : 1);
    float scale = t->init == ES_INIT_WEIGHT ? 1.0f / sqrtf((float)t->d0) : 0.1f;
    for (int j = 0; j < count; j++) {
      float u = synth_uniform(&x);
      w[j] = t->init == ES_INIT_LN_WEIGHT ? 1.0f + 0.1f * u : scale * u;
    }
    *t->dst = w;
  }

  /* All-pass: dec.k.fc2 weight 0 and bias atanh(2/3) on the real tap-0 entries (index j < W_k of the
     band's [re: T x W][im: T x W] layout), 0 elsewhere. Then every coefficient is
     1.5 tanh(atanh(2/3)) = 1 on tap 0 and 0 on the other taps and the imaginary parts: Y = X.
     The fields are const for the model's readers; the arena belongs to us, so we may write it. */
  if (allpass) {
    float c = (float)atanh(2.0 / 3.0);           /* 0.80471896 */
    for (int k = 0; k < 16; k++) {
      int W = m->band_width[k], n_coef = 2 * W * m->band_taps[k];
      float *w2 = (float *)m->dec[k].fc2.w, *b2 = (float *)m->dec[k].fc2.b;
      memset(w2, 0, sizeof(float) * 128 * (size_t)n_coef);
      for (int j = 0; j < n_coef; j++) b2[j] = j < W ? c : 0.0f;
    }
  }

  /* Hashes like a loaded model: the config hash, and SHA-256 of the weight bytes as the data hash. */
  es_sha256_hex(ES_CONFIG_CANON, strlen(ES_CONFIG_CANON), m->config_hash);
  es_sha256_hex(m->arena, m->arena_bytes, m->data_hash);
  free(list);
  free(off);
  *out = m;
  return ES_OK;
}

void es_model_free(es_model *m) {
  if (!m) return;
  free(m->arena);
  free(m);
}

const char *es_model_config_hash(const es_model *m) { return m ? m->config_hash : NULL; }
const char *es_model_data_hash(const es_model *m) { return m ? m->data_hash : NULL; }

/* Algorithmic latency of the model at 16 kHz (contract 4): 128 samples = 8 ms. */
int es_model_latency_samples(const es_model *m) {
  (void)m;
  return 128;
}

/* ------------------------------------------------------------------------------------------------
 * Package E-1: loading a real .esw (20-engine 4.3).
 *
 * es_model_load copies the whole data region of the file into one 64-byte aligned arena with a single
 * memcpy. Every tensor keeps its 64-aligned offset, so binding a tensor is just "pointer = arena +
 * offset". After that the caller may free the file buffer (contract 3.1).
 * ---------------------------------------------------------------------------------------------- */

/* Find array `name` in the container, require dtype f32 and shape [d0] (d1 == 0) or [d0, d1], set
   *dst = arena + offset and mark the array used. Returns 0, or writes err and returns -1. The error
   names the tensor, for example "array blocks.2.t.w_hh: shape [128,256], expected [128,512]". */
static int bind(es_model *m, const es_container *ct, unsigned char *used, const char *name, int d0, int d1,
                const float **dst, char *err, size_t errlen) {
  const es_array *a = es_container_find(ct, name);
  if (!a) { fail(err, errlen, ES_ERR_FORMAT, "array %s: missing", name); return -1; }
  if (a->dtype != ES_DT_F32) { fail(err, errlen, ES_ERR_FORMAT, "array %s: dtype is not f32", name); return -1; }
  int want_ndim = d1 > 0 ? 2 : 1;
  int same = a->ndim == want_ndim && a->shape[0] == d0 && (want_ndim == 1 || a->shape[1] == d1);
  if (!same) {
    char got[64] = "[", piece[16];
    for (int d = 0; d < a->ndim; d++) {
      snprintf(piece, sizeof(piece), d ? ",%d" : "%d", a->shape[d]);
      strncat(got, piece, sizeof(got) - strlen(got) - 2);
    }
    strncat(got, "]", sizeof(got) - strlen(got) - 1);
    if (d1 > 0) fail(err, errlen, ES_ERR_FORMAT, "array %s: shape %s, expected [%d,%d]", name, got, d0, d1);
    else        fail(err, errlen, ES_ERR_FORMAT, "array %s: shape %s, expected [%d]", name, got, d0);
    return -1;
  }
  *dst = (const float *)((const char *)m->arena + a->offset);
  used[a - ct->arrays] = 1;
  return 0;
}

/* The loader of 20-engine 4.1 and 4.3: container checks 1 to 10 (magic ESW1), one arena copy, then
   all 270 tensors bound by name in contract 3.2 order; any array that is not one of them is an
   extra tensor and fails the load. */
es_status es_model_load(const void *esw, size_t nbytes, es_model **out, char *err, size_t errlen) {
  if (err && errlen > 0) err[0] = '\0';
  if (!out) return fail(err, errlen, ES_ERR_ARG, "argument: out is NULL");
  *out = NULL;
  if (!esw) return fail(err, errlen, ES_ERR_ARG, "argument: buffer is NULL");

  es_container ct;
  es_status st = es_container_open(esw, nbytes, "ESW1", &ct, err, errlen);
  if (st != ES_OK) return st;

  es_model *m = calloc(1, sizeof(es_model));
  es_tensor *list = malloc(sizeof(es_tensor) * ES_N_TENSORS);
  unsigned char *used = calloc((size_t)(ct.n_arrays > 0 ? ct.n_arrays : 1), 1);
  void *arena = NULL;
  if (m && list && used && posix_memalign(&arena, 64, ct.data_bytes > 0 ? ct.data_bytes : 64) != 0) arena = NULL;
  if (!arena) {
    free(m); free(list); free(used);
    es_container_close(&ct);
    return fail(err, errlen, ES_ERR_NOMEM, "model: out of memory for the %zu-byte arena", ct.data_bytes);
  }
  memcpy(arena, ct.base + ct.data_start, ct.data_bytes);   /* one copy keeps every 64-byte offset */
  m->arena = arena;
  m->arena_bytes = ct.data_bytes;
  model_fixed_tables(m);
  memcpy(m->config_hash, ct.config_hash, sizeof(m->config_hash));   /* verified by checks 7 and 9 */
  memcpy(m->data_hash, ct.data_hash, sizeof(m->data_hash));

  /* Bind every tensor of the list (the same list the synthetic model fills). */
  st = ES_OK;
  int n = es_model_tensors(m, list);
  for (int i = 0; i < n && st == ES_OK; i++)
    if (bind(m, &ct, used, list[i].name, list[i].d0, list[i].d1, list[i].dst, err, errlen) != 0) st = ES_ERR_FORMAT;

  /* Anything left over is an extra tensor; the file must hold exactly the 270. */
  for (int i = 0; i < ct.n_arrays && st == ES_OK; i++)
    if (!used[i]) st = fail(err, errlen, ES_ERR_FORMAT, "array %s: unexpected tensor", ct.arrays[i].name);
  if (st == ES_OK && (n != ES_N_TENSORS || ct.n_arrays != ES_N_TENSORS))
    st = fail(err, errlen, ES_ERR_FORMAT, "arrays: %d tensors, expected %d", ct.n_arrays, ES_N_TENSORS);

  free(list);
  free(used);
  es_container_close(&ct);                         /* the model keeps no pointer into the file buffer */
  if (st != ES_OK) {
    es_model_free(m);
    return st;
  }
  *out = m;
  return ES_OK;
}

/* Whole file into one malloc'd buffer (non-RT). Files above 2 GB are refused (no export is near that). */
es_status es_read_file(const char *path, void **buf, size_t *n) {
  if (!path || !buf || !n) return ES_ERR_ARG;
  *buf = NULL;
  *n = 0;
  FILE *f = fopen(path, "rb");
  if (!f) return ES_ERR_ARG;
  long size = -1;
  if (fseek(f, 0, SEEK_END) == 0) size = ftell(f);
  if (size < 0 || size > 0x7fffffffL || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return ES_ERR_ARG; }
  void *b = malloc(size > 0 ? (size_t)size : 1);
  if (!b) { fclose(f); return ES_ERR_NOMEM; }
  size_t got = fread(b, 1, (size_t)size, f);
  fclose(f);
  if (got != (size_t)size) { free(b); return ES_ERR_ARG; }
  *buf = b;
  *n = (size_t)size;
  return ES_OK;
}

es_status es_model_load_file(const char *path, es_model **out, char *err, size_t errlen) {
  void *buf = NULL;
  size_t n = 0;
  if (out) *out = NULL;
  es_status st = es_read_file(path, &buf, &n);
  if (st != ES_OK) return fail(err, errlen, st, "file: cannot read %s", path ? path : "(null)");
  st = es_model_load(buf, n, out, err, errlen);
  free(buf);                                       /* allowed: the model copied what it needs */
  return st;
}
