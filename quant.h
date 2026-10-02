// cpp/quant.h — INT8 quantization (per-channel) + int8 matvec.
#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include "matmul.h"

// Per-channel INT8 tensor: one scale per output row.
//   W_int8[j,i] in [-127, 127]
//   W_fp32[j,i] ? W_int8[j,i] * scales[j]
struct Int8Tensor {
    std::vector<int8_t> data;    // [rows * cols]
    std::vector<float>  scales;  // [rows]
    int rows = 0;
    int cols = 0;

    size_t bytes() const {
        return data.size() + scales.size() * sizeof(float);
    }

    bool empty() const { return data.empty(); }
};

// Quantize an FP32 matrix [rows, cols] (row-major) to per-channel INT8.
inline Int8Tensor quantize_per_channel(const std::vector<float>& fp32,
    int rows, int cols) {
    Int8Tensor t;
    t.rows = rows;
    t.cols = cols;
    t.data.resize(static_cast<size_t>(rows) * cols);
    t.scales.resize(rows);

    for (int j = 0; j < rows; ++j) {
        const float* row = fp32.data() + static_cast<size_t>(j) * cols;

        // Row range ? scale.
        float mx = 0.0f;
        for (int i = 0; i < cols; ++i) {
            const float a = std::fabs(row[i]);
            if (a > mx) mx = a;
        }
        float scale = mx / 127.0f;
        if (scale < 1e-9f) scale = 1e-9f;
        t.scales[j] = scale;

        // Scale ? int8.
        int8_t* dst = t.data.data() + static_cast<size_t>(j) * cols;
        const float inv = 1.0f / scale;
        for (int i = 0; i < cols; ++i) {
            int32_t q = static_cast<int32_t>(std::lround(row[i] * inv));
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            dst[i] = static_cast<int8_t>(q);
        }
    }
    return t;
}

// Vectorized int8 dot product.
//   SSE2  : 16 int8 per iteration (works everywhere on x86_64)
//   AVX2  : 32 int8 per iteration (needs Haswell+; Pentium/Celeron
//           Kaby/Coffee Lake and older CPUs lack AVX2 and will use SSE2)
//
// NOTE: 256-bit *integer* intrinsics (_mm256_srai_epi16, _mm256_madd_epi16,
// _mm256_unpacklo_epi8, _mm256_add_epi32, ...) belong to AVX2, NOT AVX1.
// That is why the gate below is `__AVX2__`, not `__AVX__`. With `-mavx`
// on an AVX1-only CPU the AVX2 branch is compiled out and the SSE2 path
// is used instead.
inline int32_t dot_int8_scalar(const int8_t* __restrict a,
    const int8_t* __restrict b, int n) {
#if defined(__AVX2__)
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(b + i));

        __m256i va_lo = _mm256_srai_epi16(
            _mm256_unpacklo_epi8(_mm256_setzero_si256(), va), 8);
        __m256i vb_lo = _mm256_srai_epi16(
            _mm256_unpacklo_epi8(_mm256_setzero_si256(), vb), 8);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va_lo, vb_lo));

        __m256i va_hi = _mm256_srai_epi16(
            _mm256_unpackhi_epi8(_mm256_setzero_si256(), va), 8);
        __m256i vb_hi = _mm256_srai_epi16(
            _mm256_unpackhi_epi8(_mm256_setzero_si256(), vb), 8);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va_hi, vb_hi));
    }

    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i s4 = _mm_add_epi32(lo, hi);
    alignas(16) int32_t buf[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(buf), s4);
    int32_t total = buf[0] + buf[1] + buf[2] + buf[3];

    for (; i < n; ++i)
        total += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    return total;
#else
    // SSE2 fallback — used on AVX1-only CPUs (Kaby/Coffee Lake Pentium,
    // Celeron, Sandy/Ivy Bridge, ...). 16 int8 per iteration.
    __m128i acc = _mm_setzero_si128();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i va = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(a + i));
        __m128i vb = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(b + i));

        __m128i va_lo = _mm_srai_epi16(
            _mm_unpacklo_epi8(_mm_setzero_si128(), va), 8);
        __m128i vb_lo = _mm_srai_epi16(
            _mm_unpacklo_epi8(_mm_setzero_si128(), vb), 8);
        acc = _mm_add_epi32(acc, _mm_madd_epi16(va_lo, vb_lo));

        __m128i va_hi = _mm_srai_epi16(
            _mm_unpackhi_epi8(_mm_setzero_si128(), va), 8);
        __m128i vb_hi = _mm_srai_epi16(
            _mm_unpackhi_epi8(_mm_setzero_si128(), vb), 8);
        acc = _mm_add_epi32(acc, _mm_madd_epi16(va_hi, vb_hi));
    }

    alignas(16) int32_t buf[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(buf), acc);
    int32_t total = buf[0] + buf[1] + buf[2] + buf[3];

    for (; i < n; ++i)
        total += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    return total;
#endif
}

// ??? Single int8 matvec ??????????????????????????????????????
//   out[j] = (sum_i x_q[i] * W_q[j,i]) * scale_x * W.scales[j] + b[j]
// x_q is produced by quantizing x with one per-tensor scale.
inline void matvec_int8(const float* __restrict x, int in_dim,
    const Int8Tensor& W,
    const float* __restrict b,
    float* __restrict out, int out_dim) {
    // Per-tensor quantize of x.
    float x_max = 0.0f;
    for (int i = 0; i < in_dim; ++i) {
        const float a = std::fabs(x[i]);
        if (a > x_max) x_max = a;
    }
    float sx = x_max / 127.0f;
    if (sx < 1e-9f) sx = 1e-9f;
    const float sx_inv = 1.0f / sx;

    // Thread-local scratch.
    static thread_local std::vector<int8_t> x_q;
    if (static_cast<int>(x_q.size()) < in_dim) x_q.resize(in_dim);

    for (int i = 0; i < in_dim; ++i) {
        int32_t q = static_cast<int32_t>(std::lround(x[i] * sx_inv));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        x_q[i] = static_cast<int8_t>(q);
    }

    const int8_t* xq = x_q.data();
    for (int j = 0; j < out_dim; ++j) {
        const int8_t* wrow = W.data.data() + static_cast<size_t>(j) * in_dim;
        const int32_t acc = dot_int8_scalar(xq, wrow, in_dim);
        out[j] = acc * sx * W.scales[j] + (b ? b[j] : 0.0f);
    }
}

// ??? Parallel int8 matvec ????????????????????????????????????
struct Int8Task {
    const int8_t* xq;
    const Int8Tensor* W;
    const float* b;
    float* out;
    int in_dim, j_begin, j_end;
    float sx;
};

inline void matvec_int8_worker(Int8Task* t) {
    const Int8Tensor& W = *t->W;
    for (int j = t->j_begin; j < t->j_end; ++j) {
        const int8_t* wrow = W.data.data() + static_cast<size_t>(j) * t->in_dim;
        const int32_t acc = dot_int8_scalar(t->xq, wrow, t->in_dim);
        t->out[j] = acc * t->sx * W.scales[j] + (t->b ? t->b[j] : 0.0f);
    }
}

inline void matvec_int8_parallel(const float* __restrict x, int in_dim,
    const Int8Tensor& W,
    const float* __restrict b,
    float* __restrict out, int out_dim,
    int nthreads = 0) {
    if (nthreads <= 0) nthreads = num_threads();
    if (out_dim < 2048 || nthreads == 1) {
        matvec_int8(x, in_dim, W, b, out, out_dim);
        return;
    }

    // Quantize x once (shared by all threads).
    float x_max = 0.0f;
    for (int i = 0; i < in_dim; ++i) {
        const float a = std::fabs(x[i]);
        if (a > x_max) x_max = a;
    }
    float sx = x_max / 127.0f;
    if (sx < 1e-9f) sx = 1e-9f;
    const float sx_inv = 1.0f / sx;

    std::vector<int8_t> x_q(in_dim);
    for (int i = 0; i < in_dim; ++i) {
        int32_t q = static_cast<int32_t>(std::lround(x[i] * sx_inv));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        x_q[i] = static_cast<int8_t>(q);
    }

    const int chunk = (out_dim + nthreads - 1) / nthreads;
    std::vector<std::thread> pool;
    std::vector<Int8Task> tasks(nthreads);
    pool.reserve(nthreads);

    for (int t = 0; t < nthreads; ++t) {
        const int jb = t * chunk;
        const int je = std::min(jb + chunk, out_dim);
        if (jb >= je) continue;
        tasks[t] = { x_q.data(), &W, b, out, in_dim, jb, je, sx };
        pool.emplace_back(matvec_int8_worker, &tasks[t]);
    }
    for (auto& th : pool) th.join();
}

// ??? Batched int8 matvec (for prefill) ???????????????????????
// X[N, in_dim] FP32 ? out[N, out_dim] FP32.
inline void matvec_int8_batch(const float* __restrict X, int N, int in_dim,
    const Int8Tensor& W,
    const float* __restrict b,
    float* __restrict out, int out_dim,
    int nthreads = 0) {
    if (nthreads <= 0) nthreads = num_threads();
    if (N == 0) return;

    // Quantize each row of X separately (per-tensor per-row scale).
    std::vector<int8_t> X_q(static_cast<size_t>(N) * in_dim);
    std::vector<float>  X_scales(N);

    for (int n = 0; n < N; ++n) {
        const float* xrow = X + static_cast<size_t>(n) * in_dim;
        float mx = 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            const float a = std::fabs(xrow[i]);
            if (a > mx) mx = a;
        }
        float sx = mx / 127.0f;
        if (sx < 1e-9f) sx = 1e-9f;
        X_scales[n] = sx;
        const float inv = 1.0f / sx;
        int8_t* dst = X_q.data() + static_cast<size_t>(n) * in_dim;
        for (int i = 0; i < in_dim; ++i) {
            int32_t q = static_cast<int32_t>(std::lround(xrow[i] * inv));
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            dst[i] = static_cast<int8_t>(q);
        }
    }

    auto worker = [&](int n_begin, int n_end) {
        for (int n = n_begin; n < n_end; ++n) {
            const int8_t* xq = X_q.data() + static_cast<size_t>(n) * in_dim;
            const float sx = X_scales[n];
            float* orow = out + static_cast<size_t>(n) * out_dim;
            for (int j = 0; j < out_dim; ++j) {
                const int8_t* wrow = W.data.data() + static_cast<size_t>(j) * in_dim;
                const int32_t acc = dot_int8_scalar(xq, wrow, in_dim);
                orow[j] = acc * sx * W.scales[j] + (b ? b[j] : 0.0f);
            }
        }
    };

    if (N >= 2 && nthreads > 1) {
        const int chunk = (N + nthreads - 1) / nthreads;
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (int t = 0; t < nthreads; ++t) {
            const int nb = t * chunk;
            const int ne = std::min(nb + chunk, N);
            if (nb >= ne) continue;
            pool.emplace_back(worker, nb, ne);
        }
        for (auto& th : pool) th.join();
    }
    else {
        worker(0, N);
    }
}