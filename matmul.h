#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#undef min
#undef max

#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <thread>
#include <algorithm>
#include <immintrin.h>

static constexpr int kMaxRopeDim = 128;
static constexpr int kMaxRopePos = 8192;

struct RopeCache {
    bool inited = false;
    float cos_tab[kMaxRopePos][kMaxRopeDim / 2] = {};
    float sin_tab[kMaxRopePos][kMaxRopeDim / 2] = {};
};

inline RopeCache& rope_cache() {
    static RopeCache c;
    return c;
}

inline void init_rope_cache(int head_dim, float theta_base) {
    RopeCache& c = rope_cache();
    if (c.inited) return;

    if (head_dim > kMaxRopeDim)
        throw std::runtime_error("head_dim exceeds kMaxRopeDim");

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
}

inline int num_threads() {
    const unsigned n = std::thread::hardware_concurrency();
    return n > 0 ? static_cast<int>(n) : 4;
}

inline float hsum_sse(__m128 v) {
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(v, shuf);
    shuf = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(1, 0, 3, 2));
    sums = _mm_add_ps(sums, shuf);
    return _mm_cvtss_f32(sums);
}

inline void rms_norm(const float* __restrict x,
    const float* __restrict gamma,
    int n, float eps, float* out) {
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

inline void rms_norm_batch(const float* __restrict X, int N, int H,
    const float* __restrict gamma, float eps,
    float* __restrict out) {
    for (int n = 0; n < N; ++n) {
        rms_norm(X + static_cast<size_t>(n) * H, gamma, H, eps,
            out + static_cast<size_t>(n) * H);
    }
}

inline void silu_inplace(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        const float v = x[i];
        x[i] = v / (1.0f + std::exp(-v));
    }
}

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

inline void apply_rope(float* vec, size_t offset, int head_dim, int position) {
    if (position < 0 || position >= kMaxRopePos)
        throw std::runtime_error("apply_rope: position out of range");

    const size_t half = static_cast<size_t>(head_dim) / 2;
    const RopeCache& c = rope_cache();
    const float* cos_row = c.cos_tab[position];
    const float* sin_row = c.sin_tab[position];
    for (size_t i = 0; i < half; ++i) {
        const float x0 = vec[offset + i];
        const float x1 = vec[offset + i + half];
        vec[offset + i] = x0 * cos_row[i] - x1 * sin_row[i];
        vec[offset + i + half] = x0 * sin_row[i] + x1 * cos_row[i];
    }
}