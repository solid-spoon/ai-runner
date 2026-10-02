// cpp/sampler.h — Token sampling with temperature, top-k, top-p and repetition penalty.
#pragma once

#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

struct SamplerConfig {
    float temperature = 0.7f;
    int   top_k = 20;
    float top_p = 0.8f;
    float repetition_penalty = 1.05f;
    int   repetition_window = 64;
};

class Sampler {
public:
    explicit Sampler(uint32_t seed = 42) : rng_(seed) {}

    // Samples a token id from logits. `logits` is not modified; a copy is
    // used internally. `recent_tokens` drives the repetition penalty.
    int sample(const float* logits, int n, const SamplerConfig& cfg,
        const std::vector<int>& recent_tokens) {
        std::vector<float> scaled(logits, logits + n);

        // Temperature scaling.
        const float inv_t = 1.0f / std::max(cfg.temperature, 1e-6f);
        for (int i = 0; i < n; ++i) scaled[i] *= inv_t;

        // Repetition penalty.
        if (cfg.repetition_penalty != 1.0f && !recent_tokens.empty()) {
            const int start = std::max(
                0, static_cast<int>(recent_tokens.size()) - cfg.repetition_window);
            for (int j = start; j < static_cast<int>(recent_tokens.size()); ++j) {
                const int t = recent_tokens[j];
                if (t < 0 || t >= n) continue;
                if (scaled[t] > 0) scaled[t] /= cfg.repetition_penalty;
                else                scaled[t] *= cfg.repetition_penalty;
            }
        }

        // Softmax.
        float mx = scaled[0];
        for (int i = 1; i < n; ++i) if (scaled[i] > mx) mx = scaled[i];

        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            scaled[i] = std::exp(scaled[i] - mx);
            sum += scaled[i];
        }
        const float inv = 1.0f / sum;
        for (int i = 0; i < n; ++i) scaled[i] *= inv;

        // Top-k filtering.
        if (cfg.top_k > 0 && cfg.top_k < n) {
            std::vector<int> idx(n);
            for (int i = 0; i < n; ++i) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + cfg.top_k, idx.end(),
                [&](int a, int b) { return scaled[a] > scaled[b]; });
            const float cutoff = scaled[idx[cfg.top_k - 1]];
            for (int i = 0; i < n; ++i) if (scaled[i] < cutoff) scaled[i] = 0.0f;
        }

        // Top-p (nucleus) filtering.
        if (cfg.top_p < 1.0f) {
            std::vector<int> idx(n);
            for (int i = 0; i < n; ++i) idx[i] = i;
            std::sort(idx.begin(), idx.end(),
                [&](int a, int b) { return scaled[a] > scaled[b]; });

            std::vector<char> keep(n, 0);
            float cum = 0.0f;
            for (int i = 0; i < n; ++i) {
                cum += scaled[idx[i]];
                keep[idx[i]] = 1;
                if (cum >= cfg.top_p) break;
            }
            for (int i = 0; i < n; ++i) if (!keep[i]) scaled[i] = 0.0f;
        }

        // Renormalize and sample.
        float total = 0.0f;
        for (int i = 0; i < n; ++i) total += scaled[i];
        if (total <= 0.0f) return 0;

        std::uniform_real_distribution<float> dist(0.0f, total);
        const float r = dist(rng_);
        float c = 0.0f;
        for (int i = 0; i < n; ++i) {
            c += scaled[i];
            if (r < c) return i;
        }
        return n - 1;
    }

    // Greedy decoding helper.
    static int argmax(const float* logits, int n) {
        int best = 0;
        for (int i = 1; i < n; ++i)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

private:
    std::mt19937 rng_;
};