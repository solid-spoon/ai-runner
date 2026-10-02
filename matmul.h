// cpp/matmul.h — SIMD and multithreaded matrix/vector primitives.
#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#undef min
#undef max

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <algorithm>

#include <emmintrin.h>   // SSE2
#include <xmmintrin.h>   // SSE

// ??? RoPE: cos/sin cache ?????????????????????????????????????
// cos/sin depend only on (position, i). For a given position, the same
// 32 pairs are reused by all 14 heads in all 24 layers. Cache them.
static constexpr int kMaxRopeDim = 128;   // Qwen2: head_dim=64, half=32
static constexpr int kMaxRopePos = 8192;

struct RopeCache {
    bool inited = false;
    float cos_tab[kMaxRopePos][kMaxRopeDim / 2];
    float sin_tab[kMaxRopePos][kMaxRopeDim / 2];
};

inline RopeCache& rope_cache() {
    static RopeCache c;
    return c;
}

inline void init_rope_cache(int head_dim, float theta_base) {
    RopeCache& c = rope_cache();
    if (c.inited) return;
    const int half = head_dim / 2;
    for (int pos = 0; pos < kMaxRopePos; ++pos) {
        for (int i = 0; i < half; ++i) {
            const float freq = std::pow(theta_base, -2.0f * i / head_dim);
            const float angle = pos * freq;
            c.cos_tab[pos][i] = std::cos(angle);
            c.sin_tab[pos][i] = std::sin(angle);
        }
    }
    c.inited = true;
    std::fprintf(stderr, "[rope] cache initialized (%d positions ? %d dims)\n",
        kMaxRopePos, half);
}

// Returns the number of hardware threads, or 4 as a fallback.
inline int num_threads() {
    const unsigned n = std::thread::hardware_concurrency();
    return n > 0 ? static_cast<int>(n) : 4;
}

// Horizontal sum of the four floats packed into an __m128 register.
inline float hsum_sse(__m128 v) {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(1, 0, 3, 2));
    sums = _mm_add_ps(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// SSE2 dot product of two float vectors of length n.
inline float dot_sse2(const float* __restrict x,
    const float* __restrict w,
    int n) {
    __m128 sum = _mm_setzero_ps();
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m128 xv = _mm_loadu_ps(x + i);
        const __m128 wv = _mm_loadu_ps(w + i);
        sum = _mm_add_ps(sum, _mm_mul_ps(xv, wv));
    }
    float s = hsum_sse(sum);
    for (; i < n; ++i) s += x[i] * w[i];
    return s;
}

// Single-threaded matvec: y = W * x + b, with W stored row-major [out_dim, in_dim].
inline void matvec(const float* __restrict x, int in_dim,
    const float* __restrict W,
    const float* __restrict b,
    float* __restrict out, int out_dim) {
    for (int j = 0; j < out_dim; ++j) {
        const float s = dot_sse2(x,
            W + static_cast<size_t>(j) * in_dim,
            in_dim);
        out[j] = s + (b ? b[j] : 0.0f);
    }
}

// Work description for a single matvec worker thread.
struct MatvecTask {
    const float* x;
    const float* W;
    const float* b;
    float* out;
    int in_dim;
    int j_begin;
    int j_end;
};

// Worker entry point: computes output rows [j_begin, j_end).
inline void matvec_worker(MatvecTask* t) {
    for (int j = t->j_begin; j < t->j_end; ++j) {
        const float s = dot_sse2(t->x,
            t->W + static_cast<size_t>(j) * t->in_dim,
            t->in_dim);
        t->out[j] = s + (t->b ? t->b[j] : 0.0f);
    }
}

// Multithreaded matvec; falls back to the single-threaded path for small outputs.
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

    const int chunk = (out_dim + nthreads - 1) / nthreads;
    std::vector<std::thread> threads;
    std::vector<MatvecTask> tasks(nthreads);
    threads.reserve(nthreads);

    for (int t = 0; t < nthreads; ++t) {
        const int begin = t * chunk;
        const int end = std::min(begin + chunk, out_dim);
        if (begin >= end) continue;
        tasks[t] = { x, W, b, out, in_dim, begin, end };
        threads.emplace_back(matvec_worker, &tasks[t]);
    }
    for (auto& th : threads) th.join();
}

// Batched matvec for prefill: X has shape [N, in_dim], output has shape [N, out_dim].
// Parallelism is across output rows, so each weight row is streamed exactly once.
inline void matvec_batch(const float* __restrict X, int N, int in_dim,
    const float* __restrict W,
    const float* __restrict b,
    float* __restrict out, int out_dim,
    int nthreads = 0) {
    if (nthreads <= 0) nthreads = num_threads();
    if (N == 0) return;

    auto worker = [&](int j_begin, int j_end) {
        for (int j = j_begin; j < j_end; ++j) {
            const float* wrow = W + static_cast<size_t>(j) * in_dim;
            const float bias = b ? b[j] : 0.0f;
            for (int n = 0; n < N; ++n) {
                const float* xrow = X + static_cast<size_t>(n) * in_dim;
                __m128 sum = _mm_setzero_ps();
                int i = 0;
                for (; i + 4 <= in_dim; i += 4) {
                    const __m128 xv = _mm_loadu_ps(xrow + i);
                    const __m128 wv = _mm_loadu_ps(wrow + i);
                    sum = _mm_add_ps(sum, _mm_mul_ps(xv, wv));
                }
                float s = bias + hsum_sse(sum);
                for (; i < in_dim; ++i) s += xrow[i] * wrow[i];
                out[static_cast<size_t>(n) * out_dim + j] = s;
            }
        }
    };

    if (out_dim >= 256 && nthreads > 1) {
        const int chunk = (out_dim + nthreads - 1) / nthreads;
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (int t = 0; t < nthreads; ++t) {
            const int jb = t * chunk;
            const int je = std::min(jb + chunk, out_dim);
            if (jb >= je) continue;
            pool.emplace_back(worker, jb, je);
        }
        for (auto& th : pool) th.join();
    }
    else {
        worker(0, out_dim);
    }
}

// RMSNorm with per-channel gamma:
//   out[i] = gamma[i] * x[i] / sqrt(mean(x^2) + eps)
inline void rms_norm(const float* __restrict x, const float* __restrict gamma,
    int n, float eps, float* __restrict out) {
    __m128 sum = _mm_setzero_ps();
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m128 xv = _mm_loadu_ps(x + i);
        sum = _mm_add_ps(sum, _mm_mul_ps(xv, xv));
    }
    float ss = hsum_sse(sum);
    for (; i < n; ++i) ss += x[i] * x[i];

    const float inv = 1.0f / std::sqrt(ss / n + eps);
    const __m128 vinv = _mm_set1_ps(inv);

    i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m128 xv = _mm_loadu_ps(x + i);
        const __m128 gv = _mm_loadu_ps(gamma + i);
        _mm_storeu_ps(out + i, _mm_mul_ps(_mm_mul_ps(xv, vinv), gv));
    }
    for (; i < n; ++i) out[i] = x[i] * inv * gamma[i];
}

// In-place SiLU activation: x = x / (1 + exp(-x)).
inline void silu_inplace(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        const float v = x[i];
        x[i] = v / (1.0f + std::exp(-v));
    }
}

// In-place numerically stable softmax.
inline void softmax_inplace(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > mx) mx = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }

    const float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

// Applies rotary position embedding (RoPE) to a single head.
// The head is laid out as [x0..x_{half-1}, x_{half}..x_{head_dim-1}].
inline void apply_rope(float* vec, int offset, int head_dim,
    int position, float /*theta_base*/) {
    const int half = head_dim / 2;
    const RopeCache& c = rope_cache();
    const float* cos_row = c.cos_tab[position];
    const float* sin_row = c.sin_tab[position];
    for (int i = 0; i < half; ++i) {
        const float x0 = vec[offset + i];
        const float x1 = vec[offset + i + half];
        vec[offset + i] = x0 * cos_row[i] - x1 * sin_row[i];
        vec[offset + i + half] = x0 * sin_row[i] + x1 * cos_row[i];
    }
}

// Batched RMSNorm applied row-wise to X of shape [N, H].
inline void rms_norm_batch(const float* __restrict X, int N, int H,
    const float* __restrict gamma, float eps,
    float* __restrict out) {
    for (int n = 0; n < N; ++n) {
        rms_norm(X + static_cast<size_t>(n) * H, gamma, H, eps,
            out + static_cast<size_t>(n) * H);
    }
}