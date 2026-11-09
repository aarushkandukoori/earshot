/* cond.c - the conditioning set: who this instance keeps (contract 2.3, 20-engine 7.4).
 *
 * es_cond_build turns voiceprints into what the hop reads every 4 ms:
 *   - film [4096]: gamma and beta for the two FiLM points, from a small generator network
 *       film = Linear(256 -> 4096)(SiLU(Linear(770 -> 256)(u))),  u = [e_t, g_t, e_o, g_o, mode]
 *     flattened with index ((p*2 + s)*16 + k)*64 + d (p = point, s = 0 gamma / 1 beta, k = band, d)
 *   - k, v [4][68][16]: attention keys and values of the token list
 *       target tokens + type_emb[0], own tokens + type_emb[1], null tokens + type_emb[2]
 *     each row through Linear(64 -> 128): the first 64 numbers are the key, the last 64 the value,
 *     head h uses numbers 16h .. 16h+15 of each. Rows past n are zero.
 * Missing pieces use the learned null vectors (Everyone mode, no own voice, no WeSpeaker).
 * Threads: any non-RT thread (allocates). Runs once per person / mode change, never per hop.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

es_cond *es_cond_new(void) {
  void *mem = NULL;
  if (posix_memalign(&mem, 64, sizeof(es_cond)) != 0) return NULL;   /* film is _Alignas(64) */
  memset(mem, 0, sizeof(es_cond));                 /* n_tokens = 0: an empty conditioning set */
  return mem;
}

void es_cond_free(es_cond *c) { free(c); }

/* A voiceprint is usable for `role` if es_enroll filled it for that role with 1..limit tokens. */
static int voiceprint_ok(const es_voiceprint *v, int role, int limit) {
  return v->role == role && v->n_tok >= 1 && v->n_tok <= limit;
}

es_status es_cond_build(const es_model *m, es_mode mode, const es_voiceprint *target,
                        const es_voiceprint *own, int use_wespk, es_cond *out) {
  /* 1. Validate: PERSON needs a target voiceprint, EVERYONE needs none; own (optional) must be OWN. */
  if (!m || !out) return ES_ERR_ARG;
  if (mode == ES_MODE_PERSON) {
    if (!target || !voiceprint_ok(target, ES_ROLE_TARGET, ES_MAX_TGT_TOKENS)) return ES_ERR_ARG;
  } else if (mode == ES_MODE_EVERYONE) {
    if (target) return ES_ERR_ARG;
  } else {
    return ES_ERR_ARG;
  }
  if (own && !voiceprint_ok(own, ES_ROLE_OWN, ES_MAX_OWN_TOKENS)) return ES_ERR_ARG;

  /* 2. WeSpeaker switch: with use_wespk = 1 but a given voiceprint that has no vector, BOTH vectors
     fall back to the nulls ("both null" is the only no-WeSpeaker state seen in training). */
  int w = use_wespk && (!target || target->has_wespk) && (!own || own->has_wespk);

  /* 3. u = [e_t (256), g_t (128), e_o (256), g_o (128), mode (2)] = 770 floats. */
  _Alignas(64) float u[770];
  memcpy(u, w && target ? target->e : m->null_e_t, sizeof(float) * 256);
  memcpy(u + 256, target ? target->g : m->null_g_t, sizeof(float) * 128);
  memcpy(u + 384, w && own ? own->e : m->null_e_o, sizeof(float) * 256);
  memcpy(u + 640, own ? own->g : m->null_g_o, sizeof(float) * 128);
  u[768] = mode == ES_MODE_PERSON ? 1.0f : 0.0f;   /* PERSON [1, 0], EVERYONE [0, 1] */
  u[769] = mode == ES_MODE_PERSON ? 0.0f : 1.0f;

  /* Token list and its key / value projection need up to 68 x (64 + 128) floats: heap, not stack. */
  float *list = malloc(sizeof(float) * ES_MAX_TOKENS * 64);
  float *kv = malloc(sizeof(float) * ES_MAX_TOKENS * 128);
  if (!list || !kv) { free(list); free(kv); return ES_ERR_NOMEM; }

  /* 4. FiLM generator: hdn = SiLU(u film.0.w + film.0.b); film = hdn film.2.w + film.2.b. */
  _Alignas(64) float hdn[256];
  es_linear(u, 1, 770, 256, m->film0.w, m->film0.b, hdn);
  for (int i = 0; i < 256; i++) hdn[i] = hdn[i] / (1.0f + expf(-hdn[i]));   /* SiLU = x * sigmoid(x) */
  es_linear(hdn, 1, 256, 4096, m->film2.w, m->film2.b, out->film);

  /* 5. Token list rows: target tokens, own tokens, then the 4 null tokens, each plus its type
     embedding; n = n_t + n_o + 4 (at most 68). */
  int n = 0;
  int n_t = target ? target->n_tok : 0, n_o = own ? own->n_tok : 0;
  for (int i = 0; i < n_t; i++, n++)
    for (int e = 0; e < 64; e++) list[n * 64 + e] = target->tokens[i][e] + m->type_emb[e];
  for (int i = 0; i < n_o; i++, n++)
    for (int e = 0; e < 64; e++) list[n * 64 + e] = own->tokens[i][e] + m->type_emb[64 + e];
  for (int i = 0; i < ES_NULL_TOKENS; i++, n++)
    for (int e = 0; e < 64; e++) list[n * 64 + e] = m->null_tok[i * 64 + e] + m->type_emb[128 + e];
  es_linear(list, n, 64, 128, m->att_kv.w, m->att_kv.b, kv);    /* [K | V] per row */

  /* 6. Scatter to head-major [h][j][16]; rows j >= n stay zero. */
  memset(out->k, 0, sizeof(out->k));
  memset(out->v, 0, sizeof(out->v));
  for (int j = 0; j < n; j++)
    for (int h = 0; h < 4; h++)
      for (int e = 0; e < 16; e++) {
        out->k[h][j][e] = kv[j * 128 + 16 * h + e];
        out->v[h][j][e] = kv[j * 128 + 64 + 16 * h + e];
      }
  out->n_tokens = n;
  out->mode = (int32_t)mode;
  free(list);
  free(kv);
  return ES_OK;
}
