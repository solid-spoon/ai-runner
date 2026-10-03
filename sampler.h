#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>
#include <cstring>

struct SamplerConfig {
    float temperature = 0.7f;
    int   top_k = 20;
    float top_p = 0.8f;
    float repetition_penalty = 1.05f;
    int   presence_penalty = 0;
    float frequency_penalty = 0.0f;
    int   repetition_window = 128;   // учитывать только последние N токенов
};

class Sampler {
    std::mt19937 rng_;
    std::vector<float> probs_;
    std::vector<int> indices_;
    std::vector<int> recent_buf_;

public:
    Sampler() {
        std::random_device rd;
        rng_.seed(rd());
    }

    void ensure_capacity(int vocab_size) {
        if ((int)probs_.size() < vocab_size) {
            probs_.resize(vocab_size);
            indices_.resize(vocab_size);
        }
    }

    void apply_penalties(float* logits, int vocab_size,
        const std::vector<int>& recent,
        const SamplerConfig& cfg) {
        if (recent.empty() || (cfg.repetition_penalty == 1.0f &&
            cfg.presence_penalty == 0 && cfg.frequency_penalty == 0.0f))
            return;

        // Окно: смотрим только на последние repetition_window токенов.
        // Это ограничивает стоимость и не подавляет слова, встречавшиеся
        // очень давно в промпте.
        const int start = (cfg.repetition_window > 0 &&
            (int)recent.size() > cfg.repetition_window)
            ? (int)recent.size() - cfg.repetition_window : 0;

        recent_buf_.assign(recent.begin() + start, recent.end());
        std::sort(recent_buf_.begin(), recent_buf_.end());

        int i = 0;
        int n = (int)recent_buf_.size();
        while (i < n) {
            int token = recent_buf_[i];
            int count = 1;
            while (i + count < n && recent_buf_[i + count] == token) count++;

            if (token >= 0 && token < vocab_size) {
                if (cfg.repetition_penalty != 1.0f) {
                    if (logits[token] > 0) logits[token] /= cfg.repetition_penalty;
                    else logits[token] *= cfg.repetition_penalty;
                }
                if (cfg.presence_penalty != 0) {
                    logits[token] -= (float)cfg.presence_penalty;
                }
                if (cfg.frequency_penalty != 0.0f) {
                    logits[token] -= cfg.frequency_penalty * (float)count;
                }
            }
            i += count;
        }
    }

    int sample(const float* logits, int vocab_size,
        const SamplerConfig& cfg,
        const std::vector<int>& recent_tokens) {
        ensure_capacity(vocab_size);

        std::memcpy(probs_.data(), logits, vocab_size * sizeof(float));
        apply_penalties(probs_.data(), vocab_size, recent_tokens, cfg);

        if (cfg.temperature <= 1e-5f) {
            return (int)(std::max_element(probs_.data(), probs_.data() + vocab_size) - probs_.data());
        }

        if (cfg.temperature != 1.0f) {
            float inv_t = 1.0f / cfg.temperature;
            for (int i = 0; i < vocab_size; ++i) probs_[i] *= inv_t;
        }

        float max_val = *std::max_element(probs_.data(), probs_.data() + vocab_size);
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            probs_[i] = std::exp(probs_[i] - max_val);
            sum += probs_[i];
        }
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < vocab_size; ++i) probs_[i] *= inv_sum;

        // Ограничиваем кандидатов, чтобы не сортировать весь словарь (150k+).
        // Если top_k не задан, берем лимит 2048, этого хватит для любого top_p.
        int max_candidates = (cfg.top_k > 0 && cfg.top_k < vocab_size) ? cfg.top_k : 2048;
        if (max_candidates > vocab_size) max_candidates = vocab_size;

        std::iota(indices_.data(), indices_.data() + vocab_size, 0);

        // O(N) поиск top-K кандидатов
        std::nth_element(indices_.data(), indices_.data() + max_candidates,
            indices_.data() + vocab_size,
            [&](int a, int b) { return probs_[a] > probs_[b]; });

        if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
            // Сортируем только отобранных кандидатов O(K log K)
            std::sort(indices_.data(), indices_.data() + max_candidates,
                [&](int a, int b) { return probs_[a] > probs_[b]; });

            float cum_prob = 0.0f;
            int cutoff = max_candidates;
            for (int i = 0; i < max_candidates; ++i) {
                cum_prob += probs_[indices_[i]];
                if (cum_prob >= cfg.top_p) {
                    cutoff = i + 1;
                    break;
                }
            }
            max_candidates = cutoff;
        }

        float new_sum = 0.0f;
        for (int i = 0; i < max_candidates; ++i) {
            new_sum += probs_[indices_[i]];
        }

        if (new_sum <= 0.0f) return indices_[0];

        std::uniform_real_distribution<float> dist(0.0f, new_sum);
        float r = dist(rng_);
        float cum = 0.0f;
        for (int i = 0; i < max_candidates; ++i) {
            cum += probs_[indices_[i]];
            if (cum >= r) return indices_[i];
        }
        return indices_[max_candidates - 1];
    }
};