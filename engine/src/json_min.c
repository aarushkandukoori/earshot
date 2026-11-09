/* json_min.c - a small strict JSON tokenizer for the container headers (20-engine 4.1, contract 3.1).
 *
 * It turns the header text into a flat array of tokens in document order (a parent always comes before
 * its children), the same idea as the jsmn library, then answers lookups by key, index or dotted path.
 * It is a recursive-descent parser: one small function per JSON construct. Nesting deeper than
 * ES_JSON_MAX_DEPTH is rejected, so the recursion depth is bounded.
 *
 * Strict on purpose (the exporter writes plain json.dumps output):
 *   - numbers follow the JSON grammar exactly (no leading zeros, no "+1", ".5", "1.", NaN, Infinity)
 *     and may be at most ES_JSON_MAX_NUM characters long;
 *   - strings may use the escapes \" \\ \/ \b \f \n \r \t; \uXXXX is rejected, as are raw control
 *     characters; strings are compared raw (byte for byte, escapes are not decoded);
 *   - exactly one value in the document, only whitespace around it.
 * Threads: any non-RT thread (es_json_parse allocates the token array with one malloc).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

typedef struct {
  const char *s;   /* the text */
  int len;         /* its length */
  int pos;         /* the next character to read */
  es_jtok *tok;    /* output tokens */
  int cap;         /* capacity of tok */
  int n;           /* tokens written */
  int depth;       /* current object / array nesting */
} parser;

static int is_digit(char c) { return c >= '0' && c <= '9'; }

static void skip_ws(parser *p) {
  while (p->pos < p->len) {
    char c = p->s[p->pos];
    if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
    p->pos++;
  }
}

/* Append a token and count it as a child of its parent. Returns its index or -1 when full. */
static int new_token(parser *p, int type, int start, int parent) {
  if (p->n >= p->cap) return -1;
  es_jtok *t = &p->tok[p->n];
  t->type = type;
  t->start = start;
  t->end = -1;
  t->size = 0;
  t->parent = parent;
  t->next = -1;
  if (parent >= 0) p->tok[parent].size++;
  return p->n++;
}

static int parse_value(parser *p, int parent);

/* A string; p->pos is on the opening quote. The token covers the bytes between the quotes. */
static int parse_string(parser *p, int parent) {
  int start = p->pos + 1, i = start;
  while (i < p->len) {
    unsigned char c = (unsigned char)p->s[i];
    if (c == '"') break;
    if (c < 0x20) return -1;                      /* raw control character */
    if (c == '\\') {
      if (i + 1 >= p->len) return -1;
      char e = p->s[i + 1];
      if (e != '"' && e != '\\' && e != '/' && e != 'b' && e != 'f' && e != 'n' && e != 'r' && e != 't')
        return -1;                                /* unknown escape, including \u */
      i += 2;
      continue;
    }
    i++;
  }
  if (i >= p->len) return -1;                     /* no closing quote */
  int t = new_token(p, ES_JSON_STRING, start, parent);
  if (t < 0) return -1;
  p->tok[t].end = i;
  p->tok[t].next = p->n;
  p->pos = i + 1;
  return t;
}

/* A number: -?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)? */
static int parse_number(parser *p, int parent) {
  int start = p->pos, i = start;
  const char *s = p->s;
  if (i < p->len && s[i] == '-') i++;
  if (i >= p->len) return -1;
  if (s[i] == '0') {
    i++;
  } else if (s[i] >= '1' && s[i] <= '9') {
    while (i < p->len && is_digit(s[i])) i++;
  } else {
    return -1;
  }
  if (i < p->len && s[i] == '.') {                /* fraction needs at least one digit */
    i++;
    if (i >= p->len || !is_digit(s[i])) return -1;
    while (i < p->len && is_digit(s[i])) i++;
  }
  if (i < p->len && (s[i] == 'e' || s[i] == 'E')) {   /* exponent needs at least one digit */
    i++;
    if (i < p->len && (s[i] == '+' || s[i] == '-')) i++;
    if (i >= p->len || !is_digit(s[i])) return -1;
    while (i < p->len && is_digit(s[i])) i++;
  }
  if (i - start > ES_JSON_MAX_NUM) return -1;     /* absurdly long number */
  int t = new_token(p, ES_JSON_NUMBER, start, parent);
  if (t < 0) return -1;
  p->tok[t].end = i;
  p->tok[t].next = p->n;
  p->pos = i;
  return t;
}

/* true, false or null. */
static int parse_literal(parser *p, int parent, const char *word, int type) {
  int n = (int)strlen(word);
  if (p->pos + n > p->len || memcmp(p->s + p->pos, word, (size_t)n) != 0) return -1;
  int t = new_token(p, type, p->pos, parent);
  if (t < 0) return -1;
  p->pos += n;
  p->tok[t].end = p->pos;
  p->tok[t].next = p->n;
  return t;
}

/* An object; p->pos is on '{'. Children are key strings; each key's single child is its value. */
static int parse_object(parser *p, int parent) {
  if (p->depth >= ES_JSON_MAX_DEPTH) return -1;
  p->depth++;
  int t = new_token(p, ES_JSON_OBJECT, p->pos, parent);
  if (t < 0) return -1;
  p->pos++;
  skip_ws(p);
  int done = 0;
  if (p->pos < p->len && p->s[p->pos] == '}') { p->pos++; done = 1; }
  while (!done) {
    skip_ws(p);
    if (p->pos >= p->len || p->s[p->pos] != '"') return -1;   /* keys must be strings */
    int key = parse_string(p, t);
    if (key < 0) return -1;
    skip_ws(p);
    if (p->pos >= p->len || p->s[p->pos] != ':') return -1;
    p->pos++;
    if (parse_value(p, key) < 0) return -1;
    p->tok[key].next = p->n;                                    /* the key's subtree ends after its value */
    skip_ws(p);
    if (p->pos >= p->len) return -1;
    if (p->s[p->pos] == ',') { p->pos++; continue; }
    if (p->s[p->pos] == '}') { p->pos++; done = 1; continue; }
    return -1;
  }
  p->tok[t].end = p->pos;
  p->tok[t].next = p->n;
  p->depth--;
  return t;
}

/* An array; p->pos is on '['. */
static int parse_array(parser *p, int parent) {
  if (p->depth >= ES_JSON_MAX_DEPTH) return -1;
  p->depth++;
  int t = new_token(p, ES_JSON_ARRAY, p->pos, parent);
  if (t < 0) return -1;
  p->pos++;
  skip_ws(p);
  int done = 0;
  if (p->pos < p->len && p->s[p->pos] == ']') { p->pos++; done = 1; }
  while (!done) {
    if (parse_value(p, t) < 0) return -1;
    skip_ws(p);
    if (p->pos >= p->len) return -1;
    if (p->s[p->pos] == ',') { p->pos++; continue; }
    if (p->s[p->pos] == ']') { p->pos++; done = 1; continue; }
    return -1;
  }
  p->tok[t].end = p->pos;
  p->tok[t].next = p->n;
  p->depth--;
  return t;
}

static int parse_value(parser *p, int parent) {
  skip_ws(p);
  if (p->pos >= p->len) return -1;
  char c = p->s[p->pos];
  if (c == '{') return parse_object(p, parent);
  if (c == '[') return parse_array(p, parent);
  if (c == '"') return parse_string(p, parent);
  if (c == '-' || is_digit(c)) return parse_number(p, parent);
  if (c == 't') return parse_literal(p, parent, "true", ES_JSON_TRUE);
  if (c == 'f') return parse_literal(p, parent, "false", ES_JSON_FALSE);
  if (c == 'n') return parse_literal(p, parent, "null", ES_JSON_NULL);
  return -1;
}

/* Parse one JSON document. On success j->tok holds j->n_tok tokens (token 0 is the root value) and must
   be released with es_json_free. On failure nothing stays allocated. Token capacity len / 2 + 16 is
   always enough: every token but the first needs at least two characters (itself and a separator). */
es_status es_json_parse(es_json *j, const char *js, uint32_t len) {
  memset(j, 0, sizeof(*j));
  if (js == NULL || len == 0 || len > 0x7fffffffu) return ES_ERR_FORMAT;
  int cap = (int)(len / 2) + 16;
  es_jtok *tok = malloc(sizeof(es_jtok) * (size_t)cap);
  if (!tok) return ES_ERR_NOMEM;
  parser p = { js, (int)len, 0, tok, cap, 0, 0 };
  int root = parse_value(&p, -1);
  skip_ws(&p);
  if (root != 0 || p.pos != p.len) {           /* a parse error, or something after the value */
    free(tok);
    return ES_ERR_FORMAT;
  }
  j->js = js;
  j->len = len;
  j->tok = tok;
  j->n_tok = p.n;
  return ES_OK;
}

void es_json_free(es_json *j) {
  free(j->tok);
  j->tok = NULL;
  j->n_tok = 0;
}

int es_json_str_len(const es_json *j, int t) {
  if (t < 0 || t >= j->n_tok || j->tok[t].type != ES_JSON_STRING) return -1;
  return j->tok[t].end - j->tok[t].start;
}

/* 1 if token t is a string whose raw bytes equal the C string s. */
int es_json_str_eq(const es_json *j, int t, const char *s) {
  int n = es_json_str_len(j, t);
  if (n < 0 || s == NULL) return 0;
  return strlen(s) == (size_t)n && memcmp(j->js + j->tok[t].start, s, (size_t)n) == 0;
}

/* Value token of `key` in object `obj`, or -1. The first key with that name wins. */
int es_json_get(const es_json *j, int obj, const char *key) {
  if (obj < 0 || obj >= j->n_tok || j->tok[obj].type != ES_JSON_OBJECT) return -1;
  int i = obj + 1;                             /* the first key */
  for (int c = 0; c < j->tok[obj].size; c++) {
    if (es_json_str_eq(j, i, key)) return i + 1;   /* the value follows its key directly */
    i = j->tok[i].next;                        /* skip the key and its value */
  }
  return -1;
}

/* Follow a dotted key path such as "meta.tolerances.out" from object obj. Each part is a key. */
int es_json_get_path(const es_json *j, int obj, const char *dotted) {
  char part[128];
  int t = obj;
  const char *s = dotted;
  while (t >= 0 && s && *s) {
    const char *dot = strchr(s, '.');
    size_t n = dot ? (size_t)(dot - s) : strlen(s);
    if (n == 0 || n >= sizeof(part)) return -1;
    memcpy(part, s, n);
    part[n] = '\0';
    t = es_json_get(j, t, part);
    s = dot ? dot + 1 : NULL;
  }
  return t;
}

/* Element i of array arr, or -1. */
int es_json_arr_at(const es_json *j, int arr, int i) {
  if (arr < 0 || arr >= j->n_tok || j->tok[arr].type != ES_JSON_ARRAY) return -1;
  if (i < 0 || i >= j->tok[arr].size) return -1;
  int t = arr + 1;
  for (int c = 0; c < i; c++) t = j->tok[t].next;
  return t;
}

/* Read number token t with strtod on a NUL-terminated copy. 0 if it is a finite number, else -1
   (wrong token type, or a value like 1e999 that overflows to infinity). strtod follows the C locale,
   which the engine and the app never change, so the decimal point is always '.'. */
int es_json_num(const es_json *j, int t, double *out) {
  if (t < 0 || t >= j->n_tok || j->tok[t].type != ES_JSON_NUMBER) return -1;
  int n = j->tok[t].end - j->tok[t].start;
  if (n <= 0 || n > ES_JSON_MAX_NUM) return -1;
  char buf[ES_JSON_MAX_NUM + 1];
  memcpy(buf, j->js + j->tok[t].start, (size_t)n);
  buf[n] = '\0';
  char *end = NULL;
  double v = strtod(buf, &end);
  if (end != buf + n || !isfinite(v)) return -1;
  *out = v;
  return 0;
}
