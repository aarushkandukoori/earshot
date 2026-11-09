// sgemm_bench.c - Accelerate single-precision matrix multiply throughput at small streaming shapes.
//
// Build:  clang -O3 -DACCELERATE_NEW_LAPACK -framework Accelerate sgemm_bench.c -o sgemm_bench
// Run:    VECLIB_MAXIMUM_THREADS=1 ./sgemm_bench          (single thread, what the audio thread gets)
//
// Each shape computes Y[M x N] = X[M x K] * W^T, with W stored [N x K] like a PyTorch Linear weight
// (cblas_sgemm RowMajor, NoTrans, Trans). For M = 1 it also times cblas_sgemv on the same data.
// The time per call is the best of 7 repetitions of a loop sized to run about 50 ms.
// GFLOPS = 2*M*N*K / time.  Matrices stay hot in cache, so this is an upper bound for one layer.

#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) { return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; }

static float *rand_buf(size_t n) {
    float *p = NULL;
    if (posix_memalign((void **)&p, 64, n * sizeof(float)) != 0) exit(1);
    for (size_t i = 0; i < n; i++) p[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    return p;
}

typedef struct {
    const char *what;
    int M, N, K;
} Shape;

static volatile float sink;

// mode 0 = sgemm (X * W^T), 1 = sgemv (W * x), 2 = sgemm with W stored [K x N] (X * W)
static double time_call(int mode, int M, int N, int K, const float *X, const float *W, float *Y) {
    int iters = 16;
    for (;;) {  // calibrate the loop to about 50 ms
        double t0 = now_s();
        for (int i = 0; i < iters; i++) {
            if (mode == 0)
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, X, K, W, K, 0.0f, Y, N);
            else if (mode == 1)
                cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K, 1.0f, W, K, X, 1, 0.0f, Y, 1);
            else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, X, K, W, N, 0.0f, Y, N);
        }
        double dt = now_s() - t0;
        if (dt > 0.05 || iters > (1 << 26)) break;
        iters *= 2;
    }
    double best = 1e30;
    for (int rep = 0; rep < 7; rep++) {
        double t0 = now_s();
        for (int i = 0; i < iters; i++) {
            if (mode == 0)
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, X, K, W, K, 0.0f, Y, N);
            else if (mode == 1)
                cblas_sgemv(CblasRowMajor, CblasNoTrans, N, K, 1.0f, W, K, X, 1, 0.0f, Y, 1);
            else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, X, K, W, N, 0.0f, Y, N);
        }
        double per = (now_s() - t0) / iters;
        if (per < best) best = per;
    }
    sink = Y[0];
    return best;
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    Shape shapes[] = {
        {"task example 8x128 * 128x256", 8, 256, 128},
        {"task example 1x256 * 256x768", 1, 768, 256},
        {"tasnet32 4ms: B->H 1x1 conv", 4, 256, 128},
        {"tasnet32 4ms: H->2B res+skip", 4, 256, 256},
        {"tasnet32 4ms: encoder L=32 -> N", 4, 256, 32},
        {"tasnet32 4ms: decoder N -> L=32", 4, 32, 256},
        {"tasnet32 8ms: B->H 1x1 conv", 8, 256, 128},
        {"tasnet32 8ms: H->2B res+skip", 8, 256, 256},
        {"tasnet64 4ms: B->H 1x1 conv", 2, 256, 128},
        {"tasnet64 4ms: H->2B res+skip", 2, 256, 256},
        {"tasnet64 8ms = tasnet32 4ms rows", 4, 256, 256},
        {"stft rnn: input 514 -> 384", 1, 384, 514},
        {"stft rnn: GRU gates 384 -> 1152", 1, 1152, 384},
        {"stft rnn: LSTM gates 384 -> 1536", 1, 1536, 384},
        {"stft rnn: out 384 -> 514", 1, 514, 384},
        {"stft rnn 8ms: input proj, 2 frames", 2, 1152, 384},
        {"stft rnn 8ms: LSTM input proj, 2 frames", 2, 1536, 384},
        {"bsrnn: time GRU gates, 16 bands", 16, 192, 64},
        {"bsrnn: time LSTM gates, 16 bands", 16, 256, 64},
        {"bsrnn: band GRU step 32 -> 96", 1, 96, 32},
        {"bsrnn: fc 16 bands 64 -> 64", 16, 64, 64},
        {"cliff probe M=2 N=1152 K=384", 2, 1152, 384},
        {"cliff probe M=3 N=1152 K=384", 3, 1152, 384},
        {"cliff probe M=4 N=1152 K=384", 4, 1152, 384},
        {"cliff probe M=8 N=1152 K=384", 8, 1152, 384},
        {"cliff probe M=16 N=1152 K=384", 16, 1152, 384},
        {"cliff probe M=2 N=512 K=384", 2, 512, 384},
        {"cliff probe M=2 N=768 K=384", 2, 768, 384},
        {"cliff probe M=2 N=1024 K=384", 2, 1024, 384},
        {"cliff probe M=2 N=384 K=514", 2, 384, 514},
        {"cliff probe M=2 N=514 K=384", 2, 514, 384},
        {"reference 64x256x256", 64, 256, 256},
        {"reference 256^3", 256, 256, 256},
        {"reference 512^3", 512, 512, 512},
        {"reference 1024^3", 1024, 1024, 1024},
    };
    int n = (int)(sizeof(shapes) / sizeof(shapes[0]));
    const char *thr = getenv("VECLIB_MAXIMUM_THREADS");
    printf("{\"meta\": {\"VECLIB_MAXIMUM_THREADS\": \"%s\"}}\n", thr ? thr : "unset");
    for (int s = 0; s < n; s++) {
        Shape sh = shapes[s];
        float *X = rand_buf((size_t)sh.M * sh.K);
        float *W = rand_buf((size_t)sh.N * sh.K);
        float *Y = rand_buf((size_t)sh.M * sh.N);
        double flop = 2.0 * sh.M * sh.N * sh.K;
        double t_nt = time_call(0, sh.M, sh.N, sh.K, X, W, Y);
        double t_nn = time_call(2, sh.M, sh.N, sh.K, X, W, Y);
        printf("{\"shape\": \"%s\", \"M\": %d, \"N\": %d, \"K\": %d, \"sgemm_xWT_us\": %.3f, \"sgemm_xWT_gflops\": %.2f, "
               "\"sgemm_xW_us\": %.3f, \"sgemm_xW_gflops\": %.2f",
               sh.what, sh.M, sh.N, sh.K, t_nt * 1e6, flop / t_nt / 1e9, t_nn * 1e6, flop / t_nn / 1e9);
        if (sh.M == 1) {
            double t_v = time_call(1, sh.M, sh.N, sh.K, X, W, Y);
            printf(", \"sgemv_us\": %.3f, \"sgemv_gflops\": %.2f", t_v * 1e6, flop / t_v / 1e9);
        }
        printf("}\n");
        fflush(stdout);
        free(X);
        free(W);
        free(Y);
    }
    return 0;
}
