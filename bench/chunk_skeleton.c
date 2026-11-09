// chunk_skeleton.c - per-chunk compute floor of each streaming candidate, written with Accelerate.
//
// Random weights, but the same layer shapes, weight memory, ring buffers, FFTs and elementwise
// passes that a real single-threaded streaming implementation would run (time-major rows, one row
// per frame). The audio it outputs means nothing; only the timing matters. It shows how fast the
// candidates could run once PyTorch's per-op overhead is gone (Swift/C on the phone).
//
// Build: clang -O3 -DACCELERATE_NEW_LAPACK -framework Accelerate chunk_skeleton.c -o chunk_skeleton
// Run:   VECLIB_MAXIMUM_THREADS=1 ./chunk_skeleton

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

static size_t g_weight_floats = 0;  // weight memory of the model being built

static float *buf(size_t n, float scale) {
    float *p = NULL;
    if (posix_memalign((void **)&p, 64, (n ? n : 1) * sizeof(float)) != 0) exit(1);
    for (size_t i = 0; i < n; i++) p[i] = scale * ((float)rand() / (float)RAND_MAX - 0.5f);
    return p;
}
static float *weights(size_t n, int fan_in) {
    g_weight_floats += n;
    return buf(n, 2.0f / sqrtf((float)fan_in));
}
static float *zeros(size_t n) { float *p = buf(n, 0.0f); return p; }

// Y[M,N] = X[M,K] * W[K,N] (+ b). Weights are stored pre-transposed [K,N]: sgemm_bench showed the
// PyTorch layout (X * W^T, CblasTrans) is about 3x slower at these sizes. For M <= 3 with a large
// weight matrix, one call per row is faster (Accelerate seems to repack W on every M >= 2 call).
static void linear(const float *X, const float *W, const float *b, float *Y, int M, int N, int K) {
    if (b)
        for (int m = 0; m < M; m++) memcpy(Y + (size_t)m * N, b, (size_t)N * sizeof(float));
    float beta = b ? 1.0f : 0.0f;
    if (M > 1 && M <= 3 && (size_t)N * K >= 98304) {
        for (int m = 0; m < M; m++)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, N, K, 1.0f, X + (size_t)m * K, K, W, N, beta,
                        Y + (size_t)m * N, N);
        return;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, X, K, W, N, beta, Y, N);
}

static void prelu(float *x, int n, float a) {
    for (int i = 0; i < n; i++) x[i] = x[i] > 0.0f ? x[i] : a * x[i];
}
static void sigmoid(float *x, int n) {
    float one = 1.0f;
    vDSP_vneg(x, 1, x, 1, (vDSP_Length)n);
    vvexpf(x, x, &n);
    vDSP_vsadd(x, 1, &one, x, 1, (vDSP_Length)n);
    vvrecf(x, x, &n);
}
static void tanh_(float *x, int n) { vvtanhf(x, x, &n); }

// cumulative layer norm over the rows of X[T,C]; st = {running sum, running sum sq, count}
static void cln_rows(float *X, int T, int C, double *st, const float *g, const float *b) {
    for (int t = 0; t < T; t++) {
        float *x = X + (size_t)t * C, s, p;
        vDSP_sve(x, 1, &s, (vDSP_Length)C);
        vDSP_svesq(x, 1, &p, (vDSP_Length)C);
        st[0] += s;
        st[1] += p;
        st[2] += C;
        float mean = (float)(st[0] / st[2]);
        float var = (float)(st[1] / st[2]) - mean * mean;
        float inv = 1.0f / sqrtf((var > 0 ? var : 0) + 1e-8f);
        for (int c = 0; c < C; c++) x[c] = (x[c] - mean) * inv * g[c] + b[c];
    }
}
static void layernorm_rows(float *X, int T, int C, const float *g, const float *b) {
    for (int t = 0; t < T; t++) {
        float *x = X + (size_t)t * C, mean, msq;
        vDSP_meanv(x, 1, &mean, (vDSP_Length)C);
        vDSP_measqv(x, 1, &msq, (vDSP_Length)C);
        float var = msq - mean * mean;
        float inv = 1.0f / sqrtf((var > 0 ? var : 0) + 1e-5f);
        for (int c = 0; c < C; c++) x[c] = (x[c] - mean) * inv * g[c] + b[c];
    }
}

// one GRU (gates r,z,n) or LSTM (gates i,f,g,o) step for R independent rows of width H.
// gx = W_ih x + b_ih (precomputed), gh = W_hh h + b_hh. h, c updated in place.
static void rnn_cell(int lstm, int R, int H, const float *gx, float *gh, float *h, float *c, float *tmp) {
    int G = lstm ? 4 : 3;
    for (int r = 0; r < R; r++) {
        const float *x = gx + (size_t)r * G * H;
        float *hh = gh + (size_t)r * G * H, *hr = h + (size_t)r * H;
        if (!lstm) {
            for (int i = 0; i < 2 * H; i++) tmp[i] = x[i] + hh[i];
            sigmoid(tmp, 2 * H);  // r, z
            for (int i = 0; i < H; i++) tmp[2 * H + i] = x[2 * H + i] + tmp[i] * hh[2 * H + i];
            tanh_(tmp + 2 * H, H);  // n
            for (int i = 0; i < H; i++) hr[i] = (1.0f - tmp[H + i]) * tmp[2 * H + i] + tmp[H + i] * hr[i];
        } else {
            float *cr = c + (size_t)r * H;
            for (int i = 0; i < 4 * H; i++) tmp[i] = x[i] + hh[i];
            sigmoid(tmp, 2 * H);          // i, f
            tanh_(tmp + 2 * H, H);        // g
            sigmoid(tmp + 3 * H, H);      // o
            for (int i = 0; i < H; i++) cr[i] = tmp[H + i] * cr[i] + tmp[i] * tmp[2 * H + i];
            memcpy(tmp, cr, (size_t)H * sizeof(float));
            tanh_(tmp, H);
            for (int i = 0; i < H; i++) hr[i] = tmp[3 * H + i] * tmp[i];
        }
    }
}

// ------------------------------------------------------------------------------------------
// Conv-TasNet TCN
// ------------------------------------------------------------------------------------------
typedef struct {
    int L, S, N, B, H, P, X, R, T, nb;
    float *enc_w, *ng, *nb_, *bott_w, *bott_b, *film_g, *film_b, *mask_w, *mask_b, *dec_w, mask_a;
    float **in_w, **in_b, **dw, **dw_b, **out_w, **out_b, **n1g, **n1b, **n2g, **n2b, *a1, *a2;
    float *hist, **ring, *dec_tail;
    int *cap, *pos, *dil;
    double *cln;
    float *patch, *w, *tmpN, *y, *z, *u, *o, *skip, *m, *contrib, *ola;
} TasNet;

static TasNet *tasnet_new(int L, int T) {
    TasNet *n = calloc(1, sizeof(TasNet));
    n->L = L; n->S = L / 2; n->N = 256; n->B = 128; n->H = 256; n->P = 3; n->X = 8; n->R = 3; n->T = T;
    int N = n->N, B = n->B, H = n->H, nb = n->X * n->R;
    n->nb = nb;
    g_weight_floats = 0;
    n->enc_w = weights((size_t)N * L, L);
    n->ng = buf(N, 1); n->nb_ = buf(N, 1);
    n->bott_w = weights((size_t)B * N, N); n->bott_b = weights(B, N);
    n->film_g = buf((size_t)n->R * B, 1); n->film_b = buf((size_t)n->R * B, 1);
    n->in_w = calloc(nb, sizeof(float *)); n->in_b = calloc(nb, sizeof(float *));
    n->dw = calloc(nb, sizeof(float *)); n->dw_b = calloc(nb, sizeof(float *));
    n->out_w = calloc(nb, sizeof(float *)); n->out_b = calloc(nb, sizeof(float *));
    n->n1g = calloc(nb, sizeof(float *)); n->n1b = calloc(nb, sizeof(float *));
    n->n2g = calloc(nb, sizeof(float *)); n->n2b = calloc(nb, sizeof(float *));
    n->a1 = buf(nb, 0.5f); n->a2 = buf(nb, 0.5f);
    n->ring = calloc(nb, sizeof(float *));
    n->cap = calloc(nb, sizeof(int)); n->pos = calloc(nb, sizeof(int)); n->dil = calloc(nb, sizeof(int));
    for (int k = 0; k < nb; k++) {
        n->in_w[k] = weights((size_t)H * B, B); n->in_b[k] = weights(H, B);
        n->dw[k] = weights((size_t)n->P * H, n->P); n->dw_b[k] = weights(H, n->P);
        n->out_w[k] = weights((size_t)2 * B * H, H); n->out_b[k] = weights(2 * B, H);
        n->n1g[k] = weights(H, 1); n->n1b[k] = weights(H, 1);
        n->n2g[k] = weights(H, 1); n->n2b[k] = weights(H, 1);
        n->dil[k] = 1 << (k % n->X);
        n->cap[k] = 2 * n->dil[k] + 1;
        n->ring[k] = zeros((size_t)n->cap[k] * H);
    }
    n->mask_w = weights((size_t)N * B, B); n->mask_b = weights(N, B); n->mask_a = 0.25f;
    n->dec_w = weights((size_t)N * L, N);
    n->hist = zeros((size_t)(L - n->S) + (size_t)T * n->S);
    n->dec_tail = zeros(L);
    n->cln = calloc((size_t)(1 + 2 * nb) * 3, sizeof(double));
    n->patch = zeros((size_t)T * L); n->w = zeros((size_t)T * N); n->tmpN = zeros((size_t)T * N);
    n->y = zeros((size_t)T * B); n->z = zeros((size_t)T * H); n->u = zeros((size_t)T * H);
    n->o = zeros((size_t)T * 2 * B); n->skip = zeros((size_t)T * B); n->m = zeros((size_t)T * N);
    n->contrib = zeros((size_t)T * L); n->ola = zeros((size_t)T * n->S + L);
    return n;
}

static void tasnet_chunk(void *ctx, const float *in, float *out) {
    TasNet *n = ctx;
    int L = n->L, S = n->S, N = n->N, B = n->B, H = n->H, T = n->T, ov = L - S;
    // encoder: overlapping frames of [previous ov samples | new T*S samples]
    memcpy(n->hist + ov, in, (size_t)T * S * sizeof(float));
    for (int j = 0; j < T; j++) memcpy(n->patch + (size_t)j * L, n->hist + (size_t)j * S, (size_t)L * sizeof(float));
    memmove(n->hist, n->hist + (size_t)T * S, (size_t)ov * sizeof(float));
    linear(n->patch, n->enc_w, NULL, n->w, T, N, L);
    for (int i = 0; i < T * N; i++) n->w[i] = n->w[i] > 0 ? n->w[i] : 0;
    memcpy(n->tmpN, n->w, (size_t)T * N * sizeof(float));
    cln_rows(n->tmpN, T, N, n->cln, n->ng, n->nb_);
    linear(n->tmpN, n->bott_w, n->bott_b, n->y, T, B, N);
    memset(n->skip, 0, (size_t)T * B * sizeof(float));
    for (int r = 0; r < n->R; r++) {
        for (int t = 0; t < T; t++)
            for (int c = 0; c < B; c++)
                n->y[t * B + c] = n->y[t * B + c] * n->film_g[r * B + c] + n->film_b[r * B + c];
        for (int x = 0; x < n->X; x++) {
            int k = r * n->X + x, d = n->dil[k], cap = n->cap[k];
            linear(n->y, n->in_w[k], n->in_b[k], n->z, T, H, B);
            prelu(n->z, T * H, n->a1[k]);
            cln_rows(n->z, T, H, n->cln + 3 * (1 + 2 * k), n->n1g[k], n->n1b[k]);
            const float *w0 = n->dw[k], *w1 = n->dw[k] + H, *w2 = n->dw[k] + 2 * H, *bb = n->dw_b[k];
            for (int t = 0; t < T; t++) {  // causal dilated depthwise conv through a ring buffer
                int p = n->pos[k];
                float *cur = n->ring[k] + (size_t)p * H;
                memcpy(cur, n->z + (size_t)t * H, (size_t)H * sizeof(float));
                const float *x1 = n->ring[k] + (size_t)((p - d + cap) % cap) * H;
                const float *x2 = n->ring[k] + (size_t)((p - 2 * d + 2 * cap) % cap) * H;
                float *uo = n->u + (size_t)t * H;
                for (int c = 0; c < H; c++) uo[c] = w0[c] * x2[c] + w1[c] * x1[c] + w2[c] * cur[c] + bb[c];
                n->pos[k] = (p + 1) % cap;
            }
            prelu(n->u, T * H, n->a2[k]);
            cln_rows(n->u, T, H, n->cln + 3 * (2 + 2 * k), n->n2g[k], n->n2b[k]);
            linear(n->u, n->out_w[k], n->out_b[k], n->o, T, 2 * B, H);
            for (int t = 0; t < T; t++) {
                vDSP_vadd(n->y + t * B, 1, n->o + (size_t)t * 2 * B, 1, n->y + t * B, 1, (vDSP_Length)B);
                vDSP_vadd(n->skip + t * B, 1, n->o + (size_t)t * 2 * B + B, 1, n->skip + t * B, 1, (vDSP_Length)B);
            }
        }
    }
    prelu(n->skip, T * B, n->mask_a);
    linear(n->skip, n->mask_w, n->mask_b, n->m, T, N, B);
    sigmoid(n->m, T * N);
    vDSP_vmul(n->m, 1, n->w, 1, n->m, 1, (vDSP_Length)T * N);
    // decoder: frame contributions [T,L] = masked [T,N] * W [N,L], then overlap-add
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, L, N, 1.0f, n->m, N, n->dec_w, L, 0.0f, n->contrib, L);
    memset(n->ola, 0, (size_t)(T * S + L) * sizeof(float));
    memcpy(n->ola, n->dec_tail, (size_t)ov * sizeof(float));
    for (int j = 0; j < T; j++)
        vDSP_vadd(n->ola + j * S, 1, n->contrib + (size_t)j * L, 1, n->ola + j * S, 1, (vDSP_Length)L);
    memcpy(out, n->ola, (size_t)T * S * sizeof(float));
    memcpy(n->dec_tail, n->ola + (size_t)T * S, (size_t)ov * sizeof(float));
}

// ------------------------------------------------------------------------------------------
// STFT front end shared by the STFT models (real FFT 512 with vDSP)
// ------------------------------------------------------------------------------------------
typedef struct {
    int nfft, hop, bins, nh, log2n;
    FFTSetup setup;
    float *win, *hist, *ola, *frame, *re, *im, *mag, *xr, *xi;
    DSPSplitComplex sc;
} Stft;

static void stft_init(Stft *s, int nfft, int hop, int nh) {
    s->nfft = nfft; s->hop = hop; s->bins = nfft / 2 + 1; s->nh = nh;
    s->log2n = (int)lrint(log2((double)nfft));
    s->setup = vDSP_create_fftsetup((vDSP_Length)s->log2n, kFFTRadix2);
    s->win = buf(nfft, 1.0f);
    for (int i = 0; i < nfft; i++) s->win[i] = sqrtf(0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / nfft));
    s->hist = zeros((size_t)(nfft - hop) + (size_t)nh * hop);
    s->ola = zeros((size_t)nfft + (size_t)nh * hop);
    s->frame = zeros(nfft);
    s->re = zeros(nfft / 2); s->im = zeros(nfft / 2);
    s->sc.realp = s->re; s->sc.imagp = s->im;
    s->mag = zeros(s->bins);
    s->xr = zeros((size_t)nh * s->bins); s->xi = zeros((size_t)nh * s->bins);
}

// analysis of frame j of the current chunk -> xr/xi row j (257 bins)
static void stft_analyze(Stft *s, int j) {
    int n = s->nfft, h = n / 2;
    vDSP_vmul(s->hist + (size_t)j * s->hop, 1, s->win, 1, s->frame, 1, (vDSP_Length)n);
    vDSP_ctoz((const DSPComplex *)s->frame, 2, &s->sc, 1, (vDSP_Length)h);
    vDSP_fft_zrip(s->setup, &s->sc, 1, (vDSP_Length)s->log2n, kFFTDirection_Forward);
    float *xr = s->xr + (size_t)j * s->bins, *xi = s->xi + (size_t)j * s->bins;
    xr[0] = s->re[0]; xi[0] = 0; xr[h] = s->im[0]; xi[h] = 0;
    memcpy(xr + 1, s->re + 1, (size_t)(h - 1) * sizeof(float));
    memcpy(xi + 1, s->im + 1, (size_t)(h - 1) * sizeof(float));
}

// power-law compression of row j into feat (2*bins)
static void compress_row(Stft *s, int j, float *feat) {
    int F = s->bins;
    float *xr = s->xr + (size_t)j * F, *xi = s->xi + (size_t)j * F, e = 1e-8f, pw = -0.35f;
    vDSP_vmma(xr, 1, xr, 1, xi, 1, xi, 1, s->mag, 1, (vDSP_Length)F);  // re^2 + im^2
    vDSP_vsadd(s->mag, 1, &e, s->mag, 1, (vDSP_Length)F);              // + eps
    vvpowsf(s->mag, &pw, s->mag, &F);                                   // ^((0.3 - 1) / 2)
    vDSP_vmul(xr, 1, s->mag, 1, feat, 1, (vDSP_Length)F);
    vDSP_vmul(xi, 1, s->mag, 1, feat + F, 1, (vDSP_Length)F);
}

// complex ratio mask on row j, inverse FFT, window, overlap-add into s->ola at frame j
static void stft_synth(Stft *s, int j, const float *mask) {
    int n = s->nfft, h = n / 2, F = s->bins;
    const float *xr = s->xr + (size_t)j * F, *xi = s->xi + (size_t)j * F, *mr = mask, *mi = mask + F;
    for (int k = 0; k < h; k++) {
        s->re[k] = xr[k] * mr[k] - xi[k] * mi[k];
        s->im[k] = xr[k] * mi[k] + xi[k] * mr[k];
    }
    s->im[0] = xr[h] * mr[h];  // Nyquist goes in imag[0] for the packed format
    vDSP_fft_zrip(s->setup, &s->sc, 1, (vDSP_Length)s->log2n, kFFTDirection_Inverse);
    vDSP_ztoc(&s->sc, 1, (DSPComplex *)s->frame, 2, (vDSP_Length)h);
    float scale = 1.0f / (2.0f * n);
    vDSP_vsmul(s->frame, 1, &scale, s->frame, 1, (vDSP_Length)n);
    vDSP_vmul(s->frame, 1, s->win, 1, s->frame, 1, (vDSP_Length)n);
    vDSP_vadd(s->ola + (size_t)j * s->hop, 1, s->frame, 1, s->ola + (size_t)j * s->hop, 1, (vDSP_Length)n);
}

static void stft_begin(Stft *s, const float *in) {
    memcpy(s->hist + (s->nfft - s->hop), in, (size_t)s->nh * s->hop * sizeof(float));
}
static void stft_end(Stft *s, float *out) {
    int done = s->nh * s->hop, keep = s->nfft - s->hop;
    memmove(s->hist, s->hist + done, (size_t)keep * sizeof(float));
    float norm = 0.25f;
    vDSP_vsmul(s->ola, 1, &norm, out, 1, (vDSP_Length)done);
    memmove(s->ola, s->ola + done, (size_t)keep * sizeof(float));
    memset(s->ola + keep, 0, (size_t)(s->nfft + done - keep) * sizeof(float));
}

// ------------------------------------------------------------------------------------------
// STFT + 2-layer GRU/LSTM 384
// ------------------------------------------------------------------------------------------
typedef struct {
    Stft s;
    int Hd, layers, lstm, G;
    float *inp_w, *inp_b, *ln_g, *ln_b, *film_g, *film_b, *o1_w, *o1_b, *o2_w, *o2_b;
    float **wih, **whh, **bih, **bhh, **h, **c;
    float *feat, *x, *gx, *gh, *tmp, *o1, *mask;
} StftRnn;

static StftRnn *stftrnn_new(int lstm, int nh) {
    StftRnn *m = calloc(1, sizeof(StftRnn));
    stft_init(&m->s, 512, 64, nh);
    int F = m->s.bins, Hd = 384, G = lstm ? 4 : 3;
    m->Hd = Hd; m->layers = 2; m->lstm = lstm; m->G = G;
    g_weight_floats = 0;
    m->inp_w = weights((size_t)Hd * 2 * F, 2 * F); m->inp_b = weights(Hd, 2 * F);
    m->ln_g = weights(Hd, 1); m->ln_b = weights(Hd, 1);
    m->film_g = buf(Hd, 1); m->film_b = buf(Hd, 1);
    m->wih = calloc(2, sizeof(float *)); m->whh = calloc(2, sizeof(float *));
    m->bih = calloc(2, sizeof(float *)); m->bhh = calloc(2, sizeof(float *));
    m->h = calloc(2, sizeof(float *)); m->c = calloc(2, sizeof(float *));
    for (int l = 0; l < 2; l++) {
        m->wih[l] = weights((size_t)G * Hd * Hd, Hd); m->whh[l] = weights((size_t)G * Hd * Hd, Hd);
        m->bih[l] = weights((size_t)G * Hd, Hd); m->bhh[l] = weights((size_t)G * Hd, Hd);
        m->h[l] = zeros(Hd); m->c[l] = zeros(Hd);
    }
    m->o1_w = weights((size_t)Hd * Hd, Hd); m->o1_b = weights(Hd, Hd);
    m->o2_w = weights((size_t)2 * F * Hd, Hd); m->o2_b = weights(2 * F, Hd);
    m->feat = zeros((size_t)nh * 2 * F); m->x = zeros((size_t)nh * Hd);
    m->gx = zeros((size_t)nh * G * Hd); m->gh = zeros((size_t)G * Hd); m->tmp = zeros((size_t)G * Hd);
    m->o1 = zeros((size_t)nh * Hd); m->mask = zeros((size_t)nh * 2 * F);
    return m;
}

static void stftrnn_chunk(void *ctx, const float *in, float *out) {
    StftRnn *m = ctx;
    Stft *s = &m->s;
    int nh = s->nh, F = s->bins, Hd = m->Hd, G = m->G;
    stft_begin(s, in);
    for (int j = 0; j < nh; j++) {
        stft_analyze(s, j);
        compress_row(s, j, m->feat + (size_t)j * 2 * F);
    }
    linear(m->feat, m->inp_w, m->inp_b, m->x, nh, Hd, 2 * F);
    layernorm_rows(m->x, nh, Hd, m->ln_g, m->ln_b);
    prelu(m->x, nh * Hd, 0.25f);
    for (int j = 0; j < nh; j++)
        for (int c = 0; c < Hd; c++) m->x[j * Hd + c] = m->x[j * Hd + c] * m->film_g[c] + m->film_b[c];
    for (int l = 0; l < m->layers; l++) {
        linear(m->x, m->wih[l], m->bih[l], m->gx, nh, G * Hd, Hd);  // input projections for all frames
        for (int j = 0; j < nh; j++) {                               // recurrence is sequential
            linear(m->h[l], m->whh[l], m->bhh[l], m->gh, 1, G * Hd, Hd);
            rnn_cell(m->lstm, 1, Hd, m->gx + (size_t)j * G * Hd, m->gh, m->h[l], m->c[l], m->tmp);
            memcpy(m->x + (size_t)j * Hd, m->h[l], (size_t)Hd * sizeof(float));
        }
    }
    linear(m->x, m->o1_w, m->o1_b, m->o1, nh, Hd, Hd);
    prelu(m->o1, nh * Hd, 0.25f);
    linear(m->o1, m->o2_w, m->o2_b, m->mask, nh, 2 * F, Hd);
    for (int j = 0; j < nh; j++) stft_synth(s, j, m->mask + (size_t)j * 2 * F);
    stft_end(s, out);
}

// ------------------------------------------------------------------------------------------
// Band-split dual-path RNN: 16 bands, D=64, 3 layers (time RNN across frames, biRNN across bands)
// ------------------------------------------------------------------------------------------
#define NB 16
static const int WIDTHS[NB] = {8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 33};

typedef struct {
    Stft s;
    int D, layers, lstm, G, start[NB];
    float *sw[NB], *sb[NB], *m1w[NB], *m1b[NB], *m2w[NB], *m2b[NB], *film_g, *film_b;
    float **tng, **tnb, **twih, **twhh, **tbih, **tbhh, **tfw, **tfb, **th, **tc;
    float **bng, **bnb, **fwih, **fwhh, **fbih, **fbhh, **bwih, **bwhh, **bbih, **bbhh, **bfw, **bfb;
    float *feat, *bandin, *z, *zn, *gx, *gh, *tmp, *y, *hf, *hb, *cf, *cb, *gxf, *gxb, *hid, *mask;
} Bsrnn;

static Bsrnn *bsrnn_new(int lstm, int nh) {
    Bsrnn *m = calloc(1, sizeof(Bsrnn));
    stft_init(&m->s, 512, 64, nh);
    int D = 64, G = lstm ? 4 : 3, L = 3, h2 = D / 2;
    m->D = D; m->layers = L; m->lstm = lstm; m->G = G;
    g_weight_floats = 0;
    int st = 0;
    for (int k = 0; k < NB; k++) {
        m->start[k] = st; st += WIDTHS[k];
        int w2 = 2 * WIDTHS[k];
        m->sw[k] = weights((size_t)D * w2, w2); m->sb[k] = weights(D, w2);
        m->m1w[k] = weights((size_t)2 * D * D, D); m->m1b[k] = weights(2 * D, D);
        m->m2w[k] = weights((size_t)w2 * 2 * D, 2 * D); m->m2b[k] = weights(w2, 2 * D);
    }
    m->film_g = buf(D, 1); m->film_b = buf(D, 1);
#define ARR(name) m->name = calloc(L, sizeof(float *))
    ARR(tng); ARR(tnb); ARR(twih); ARR(twhh); ARR(tbih); ARR(tbhh); ARR(tfw); ARR(tfb); ARR(th); ARR(tc);
    ARR(bng); ARR(bnb); ARR(fwih); ARR(fwhh); ARR(fbih); ARR(fbhh); ARR(bwih); ARR(bwhh); ARR(bbih); ARR(bbhh);
    ARR(bfw); ARR(bfb);
    for (int l = 0; l < L; l++) {
        m->tng[l] = weights(D, 1); m->tnb[l] = weights(D, 1);
        m->twih[l] = weights((size_t)G * D * D, D); m->twhh[l] = weights((size_t)G * D * D, D);
        m->tbih[l] = weights((size_t)G * D, D); m->tbhh[l] = weights((size_t)G * D, D);
        m->tfw[l] = weights((size_t)D * D, D); m->tfb[l] = weights(D, D);
        m->th[l] = zeros((size_t)NB * D); m->tc[l] = zeros((size_t)NB * D);
        m->bng[l] = weights(D, 1); m->bnb[l] = weights(D, 1);
        m->fwih[l] = weights((size_t)G * h2 * D, D); m->fwhh[l] = weights((size_t)G * h2 * h2, h2);
        m->fbih[l] = weights((size_t)G * h2, D); m->fbhh[l] = weights((size_t)G * h2, h2);
        m->bwih[l] = weights((size_t)G * h2 * D, D); m->bwhh[l] = weights((size_t)G * h2 * h2, h2);
        m->bbih[l] = weights((size_t)G * h2, D); m->bbhh[l] = weights((size_t)G * h2, h2);
        m->bfw[l] = weights((size_t)D * D, D); m->bfb[l] = weights(D, D);
    }
    int F = m->s.bins;
    m->feat = zeros(2 * F); m->bandin = zeros(2 * 33);
    m->z = zeros((size_t)NB * D); m->zn = zeros((size_t)NB * D);
    m->gx = zeros((size_t)NB * G * D); m->gh = zeros((size_t)NB * G * D); m->tmp = zeros((size_t)G * D);
    m->y = zeros((size_t)NB * D); m->hf = zeros(h2); m->hb = zeros(h2); m->cf = zeros(h2); m->cb = zeros(h2);
    m->gxf = zeros((size_t)NB * G * h2); m->gxb = zeros((size_t)NB * G * h2);
    m->hid = zeros(2 * D); m->mask = zeros(2 * F);
    return m;
}

static void bsrnn_chunk(void *ctx, const float *in, float *out) {
    Bsrnn *m = ctx;
    Stft *s = &m->s;
    int nh = s->nh, F = s->bins, D = m->D, G = m->G, h2 = D / 2;
    stft_begin(s, in);
    for (int j = 0; j < nh; j++) {
        stft_analyze(s, j);
        compress_row(s, j, m->feat);
        for (int k = 0; k < NB; k++) {  // band split: per-band linear (2*width -> D) + FiLM
            int w = WIDTHS[k], a = m->start[k];
            memcpy(m->bandin, m->feat + a, (size_t)w * sizeof(float));
            memcpy(m->bandin + w, m->feat + F + a, (size_t)w * sizeof(float));
            linear(m->bandin, m->sw[k], m->sb[k], m->z + (size_t)k * D, 1, D, 2 * w);
            for (int c = 0; c < D; c++) m->z[k * D + c] = m->z[k * D + c] * m->film_g[c] + m->film_b[c];
        }
        for (int l = 0; l < m->layers; l++) {
            // time path: one RNN step for the 16 bands at once (shared weights, 16 rows)
            memcpy(m->zn, m->z, (size_t)NB * D * sizeof(float));
            layernorm_rows(m->zn, NB, D, m->tng[l], m->tnb[l]);
            linear(m->zn, m->twih[l], m->tbih[l], m->gx, NB, G * D, D);
            linear(m->th[l], m->twhh[l], m->tbhh[l], m->gh, NB, G * D, D);
            rnn_cell(m->lstm, NB, D, m->gx, m->gh, m->th[l], m->tc[l], m->tmp);
            linear(m->th[l], m->tfw[l], m->tfb[l], m->y, NB, D, D);
            vDSP_vadd(m->z, 1, m->y, 1, m->z, 1, (vDSP_Length)NB * D);
            // band path: bidirectional RNN over the 16 bands of this frame
            memcpy(m->zn, m->z, (size_t)NB * D * sizeof(float));
            layernorm_rows(m->zn, NB, D, m->bng[l], m->bnb[l]);
            linear(m->zn, m->fwih[l], m->fbih[l], m->gxf, NB, G * h2, D);
            linear(m->zn, m->bwih[l], m->bbih[l], m->gxb, NB, G * h2, D);
            memset(m->hf, 0, (size_t)h2 * 4); memset(m->cf, 0, (size_t)h2 * 4);
            memset(m->hb, 0, (size_t)h2 * 4); memset(m->cb, 0, (size_t)h2 * 4);
            for (int k = 0; k < NB; k++) {
                linear(m->hf, m->fwhh[l], m->fbhh[l], m->gh, 1, G * h2, h2);
                rnn_cell(m->lstm, 1, h2, m->gxf + (size_t)k * G * h2, m->gh, m->hf, m->cf, m->tmp);
                memcpy(m->y + (size_t)k * D, m->hf, (size_t)h2 * sizeof(float));
                int kb = NB - 1 - k;
                linear(m->hb, m->bwhh[l], m->bbhh[l], m->gh, 1, G * h2, h2);
                rnn_cell(m->lstm, 1, h2, m->gxb + (size_t)kb * G * h2, m->gh, m->hb, m->cb, m->tmp);
                memcpy(m->y + (size_t)kb * D + h2, m->hb, (size_t)h2 * sizeof(float));
            }
            linear(m->y, m->bfw[l], m->bfb[l], m->zn, NB, D, D);
            vDSP_vadd(m->z, 1, m->zn, 1, m->z, 1, (vDSP_Length)NB * D);
        }
        for (int k = 0; k < NB; k++) {  // per-band mask MLP -> CRM for the band's bins
            int w = WIDTHS[k], a = m->start[k];
            linear(m->z + (size_t)k * D, m->m1w[k], m->m1b[k], m->hid, 1, 2 * D, D);
            tanh_(m->hid, 2 * D);
            linear(m->hid, m->m2w[k], m->m2b[k], m->bandin, 1, 2 * w, 2 * D);
            memcpy(m->mask + a, m->bandin, (size_t)w * sizeof(float));
            memcpy(m->mask + F + a, m->bandin + w, (size_t)w * sizeof(float));
        }
        stft_synth(s, j, m->mask);
    }
    stft_end(s, out);
}

// ------------------------------------------------------------------------------------------

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

static void bench(const char *name, int chunk, double macs_per_chunk, void (*fn)(void *, const float *, float *),
                  void *ctx, size_t weight_floats) {
    float *in = buf((size_t)chunk, 0.2f), *out = zeros((size_t)chunk + 1024);
    for (int i = 0; i < 500; i++) fn(ctx, in, out);
    int n = 20000;
    uint64_t *ts = malloc((size_t)n * sizeof(uint64_t));
    uint64_t t_all = now_ns();
    for (int i = 0; i < n; i++) {
        uint64_t t0 = now_ns();
        fn(ctx, in, out);
        ts[i] = now_ns() - t0;
    }
    double total = (double)(now_ns() - t_all) * 1e-3 / n;
    qsort(ts, (size_t)n, sizeof(uint64_t), cmp_u64);
    double med = ts[n / 2] * 1e-3, budget = chunk / 16.0;  // us; 16 samples per ms
    printf("{\"model\": \"%s\", \"chunk_samples\": %d, \"chunk_ms\": %.1f, \"weights_MB\": %.2f, "
           "\"macs_per_chunk\": %.0f, \"median_us\": %.2f, \"mean_us\": %.2f, \"p99_us\": %.2f, "
           "\"p999_us\": %.2f, \"max_us\": %.2f, \"rtf_median\": %.4f, \"gflops\": %.2f}\n",
           name, chunk, chunk / 16.0, weight_floats * 4.0 / 1e6, macs_per_chunk, med, total,
           ts[(int)(n * 0.99)] * 1e-3, ts[(int)(n * 0.999)] * 1e-3, ts[n - 1] * 1e-3, med / (budget * 1e3),
           2.0 * macs_per_chunk / (med * 1e3));
    fflush(stdout);
    free(ts);
}

int main(void) {
    srand(1);
    const char *thr = getenv("VECLIB_MAXIMUM_THREADS");
    printf("{\"meta\": {\"VECLIB_MAXIMUM_THREADS\": \"%s\"}}\n", thr ? thr : "unset");
    int chunks[2] = {64, 128};
    for (int ci = 0; ci < 2; ci++) {
        int c = chunks[ci];
        TasNet *t32 = tasnet_new(32, c / 16);
        bench("tasnet32", c, 2459648.0 * c / 16.0, tasnet_chunk, t32, g_weight_floats);
        TasNet *t64 = tasnet_new(64, c / 32);
        bench("tasnet64", c, 2476032.0 * c / 32.0, tasnet_chunk, t64, g_weight_floats);
        StftRnn *gru = stftrnn_new(0, c / 64);
        bench("stftgru", c, 2311680.0 * c / 64.0, stftrnn_chunk, gru, g_weight_floats);
        StftRnn *lstm = stftrnn_new(1, c / 64);
        bench("stftlstm", c, 2901504.0 * c / 64.0, stftrnn_chunk, lstm, g_weight_floats);
        Bsrnn *bg = bsrnn_new(0, c / 64);
        bench("bsrnn", c, 2687360.0 * c / 64.0, bsrnn_chunk, bg, g_weight_floats);
        Bsrnn *bl = bsrnn_new(1, c / 64);
        bench("bsrnnlstm", c, 3375488.0 * c / 64.0, bsrnn_chunk, bl, g_weight_floats);
    }
    return 0;
}
