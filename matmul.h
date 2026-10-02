// cpp/matmul.h — AVX2 + многопоточный matvec
#pragma once

// Обязательно НА САМОМ ВЕРХУ, до всех инклудов!
#ifndef NOMINMAX
#define NOMINMAX
#endif
#undef min
#undef max

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>      // <-- Этот парень неявно тянет за собой куски Windows API
#include <algorithm>
#include <atomic>
#include <windows.h>
#include <iostream>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

// cpp/matmul.h — SSE2 matvec (работает на любом x64 CPU)
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <algorithm>

#include <emmintrin.h>   // SSE2
#include <xmmintrin.h>   // SSE

// ??? Число потоков ??????????????????????????????????????????
inline int num_threads() {
    unsigned n = std::thread::hardware_concurrency();
    return n > 0 ? (int)n : 4;
}

// ??? Horizontal sum of __m128 (SSE2) ????????????????????????
inline float hsum_sse(__m128 v) {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(1, 0, 3, 2));
    sums = _mm_add_ps(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// ??? SSE2 dot product ???????????????????????????????????????
inline float dot_sse2(const float* __restrict x, const float* __restrict w, int n) {
    __m128 sum = _mm_setzero_ps();
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 xv = _mm_loadu_ps(x + i);
        __m128 wv = _mm_loadu_ps(w + i);
        sum = _mm_add_ps(sum, _mm_mul_ps(xv, wv));
    }
    float s = hsum_sse(sum);
    for (; i < n; i++) s += x[i] * w[i];
    return s;
}

// ??? Single matvec: y[j] = sum_i x[i] * W[j*in + i] + b[j] ??
inline void matvec(const float* __restrict x, int in_dim,
    const float* __restrict W,
    const float* __restrict b,
    float* __restrict out, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        float s = dot_sse2(x, W + (size_t)j * in_dim, in_dim);
        out[j] = s + (b ? b[j] : 0.0f);
    }
}

// ??? Parallel matvec ????????????????????????????????????????
struct MatvecTask {
    const float* x;
    const float* W;
    const float* b;
    float* out;
    int in_dim;
    int j_begin, j_end;
};

inline void matvec_worker(MatvecTask* t) {
    for (int j = t->j_begin; j < t->j_end; j++) {
        float s = dot_sse2(t->x, t->W + (size_t)j * t->in_dim, t->in_dim);
        t->out[j] = s + (t->b ? t->b[j] : 0.0f);
    }
}

inline void matvec_parallel(const float* __restrict x, int in_dim,
    const float* __restrict W,
    const float* __restrict b,
    float* __restrict out, int out_dim,
    int nthreads = 0) {
    if (nthreads <= 0) nthreads = num_threads();
    if (out_dim < 2048 || nthreads == 1) {
        matvec(x, in_dim, W, b, out, out_dim);
        return;
    }

    int chunk = (out_dim + nthreads - 1) / nthreads;
    std::vector<std::thread> threads;
    std::vector<MatvecTask> tasks(nthreads);
    threads.reserve(nthreads);

    for (int t = 0; t < nthreads; t++) {
        int begin = t * chunk;
        int end = std::min(begin + chunk, out_dim);
        if (begin >= end) continue;
        tasks[t] = { x, W, b, out, in_dim, begin, end };
        threads.emplace_back(matvec_worker, &tasks[t]);
    }
    for (auto& th : threads) th.join();
}

// ??? Batched: X[N, in], out[N, out] ?????????????????????????
inline void matvec_batch(const float* __restrict X, int N, int in_dim,
    const float* __restrict W,
    const float* __restrict b,
    float* __restrict out, int out_dim,
    int nthreads = 0) {
    if (nthreads <= 0) nthreads = num_threads();
    if (N == 0) return;

    if (N >= 2 && out_dim >= 64) {
        int chunk = (N + nthreads - 1) / nthreads;
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (int t = 0; t < nthreads; t++) {
            int n_begin = t * chunk;
            int n_end = std::min(n_begin + chunk, N);
            if (n_begin >= n_end) continue;
            pool.emplace_back([&, n_begin, n_end]() {
                for (int n = n_begin; n < n_end; n++) {
                    const float* x = X + (size_t)n * in_dim;
                    float* y = out + (size_t)n * out_dim;
                    for (int j = 0; j < out_dim; j++) {
                        float s = dot_sse2(x, W + (size_t)j * in_dim, in_dim);
                        y[j] = s + (b ? b[j] : 0.0f);
                    }
                }
                });
        }
        for (auto& th : pool) th.join();
    }
    else {
        for (int n = 0; n < N; n++) {
            matvec_parallel(X + (size_t)n * in_dim, in_dim,
                W, b, out + (size_t)n * out_dim, out_dim, nthreads);
        }
    }
}

// ??? RMSNorm (SSE2) ?????????????????????????????????????????
inline void rms_norm(const float* __restrict x, const float* __restrict gamma,
    int n, float eps, float* __restrict out) {
    __m128 sum = _mm_setzero_ps();
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 xv = _mm_loadu_ps(x + i);
        sum = _mm_add_ps(sum, _mm_mul_ps(xv, xv));
    }
    float ss = hsum_sse(sum);
    for (; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / std::sqrt(ss / n + eps);
    __m128 vinv = _mm_set1_ps(inv);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 xv = _mm_loadu_ps(x + i);
        __m128 gv = _mm_loadu_ps(gamma + i);
        _mm_storeu_ps(out + i, _mm_mul_ps(_mm_mul_ps(xv, vinv), gv));
    }
    for (; i < n; i++) out[i] = x[i] * inv * gamma[i];
}

// ??? SiLU in place ??????????????????????????????????????????
inline void silu_inplace(float* x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + std::exp(-v));
    }
}

// ??? Softmax in place ???????????????????????????????????????
inline void softmax_inplace(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = std::exp(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

// ??? RoPE ???????????????????????????????????????????????????
inline void apply_rope(float* vec, int offset, int head_dim,
    int position, float theta_base) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = std::pow(theta_base, -2.0f * i / head_dim);
        float angle = position * freq;
        float c = std::cos(angle), s = std::sin(angle);
        float x0 = vec[offset + i];
        float x1 = vec[offset + i + half];
        vec[offset + i] = x0 * c - x1 * s;
        vec[offset + i + half] = x0 * s + x1 * c;
    }
}