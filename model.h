#pragma once
#include "st.h"
#include "matmul.h"
#include "quant.h"
#include "config.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>

inline constexpr int kMaxPrefill = 1024;
inline constexpr int kMaxContext = 8192;

class QwenModel {
public:
    ModelConfig cfg;
    int H, NH, NKV, NL, HD, KVD, Q_DIM, INTER, VOCAB, GROUPS;
    float RMS_EPS;

    Int8Tensor embed;
    Int8Tensor lm_head;
    std::vector<float> finalNorm;

    struct Layer {
        std::vector<float> inputNorm, postNorm;
        std::vector<float> q_norm, k_norm;
        Int8Tensor q_w, k_w, v_w, o_w;
        std::vector<float> q_b, k_b, v_b;
        Int8Tensor gate_w, up_w, down_w;
    };
    std::vector<Layer> layers;

    std::vector<float> kv_k_flat, kv_v_flat;
    int kv_len_ = 0;

    // single-token scratch
    std::vector<float> x, xNorm, q, k, v, attnOut, oProj, gate, up, mlp;
    std::vector<float> logits, scores;
    // batch scratch
    std::vector<float> Xb, Xnb, Qb, Kb, Vb, AttnOutb, OProjb;
    std::vector<float> Gateb, Upb, Mlb, scores_buf;

    explicit QwenModel(const SafeTensors& st, const ModelConfig& mc) : cfg(mc) {
        H = cfg.hidden_size;
        NH = cfg.num_attention_heads;
        NKV = cfg.effective_kv_heads();
        HD = cfg.effective_head_dim();
        NL = cfg.num_hidden_layers;
        KVD = NKV * HD;
        Q_DIM = NH * HD;
        INTER = cfg.intermediate_size;
        VOCAB = cfg.vocab_size;
        RMS_EPS = cfg.rms_norm_eps;
        GROUPS = NH / NKV;

        cfg.use_qk_norm = st.has("model.layers.0.self_attn.q_norm.weight");
        const bool has_bias = st.has("model.layers.0.self_attn.q_proj.bias");
        const bool has_lm_head = st.has("lm_head.weight");
        const bool tied = !has_lm_head;

        std::fprintf(stderr,
            "[model] %s H=%d NH=%d NKV=%d HD=%d NL=%d INTER=%d VOCAB=%d "
            "qk_norm=%d bias=%d tied=%d\n",
            cfg.model_type.c_str(), H, NH, NKV, HD, NL, INTER, VOCAB,
            cfg.use_qk_norm ? 1 : 0, has_bias ? 1 : 0, tied ? 1 : 0);

        auto require = [&](const std::string& name) {
            if (!st.has(name)) throw std::runtime_error("missing tensor: " + name);
        };
        auto check_shape = [&](const std::string& name, int r, int c) {
            const TensorInfo& t = st[name];
            if ((int)t.shape.size() != 2 || (int)t.shape[0] != r || (int)t.shape[1] != c)
                throw std::runtime_error("tensor '" + name + "' shape mismatch");
        };
        auto load = [&](const std::string& name) {
            require(name);
            return SafeTensors::to_f32(st[name]);
        };
        auto load_int8 = [&](const std::string& name, int rows, int cols) {
            Int8Tensor t; t.rows = rows; t.cols = cols;
            const TensorInfo& w = st[name];
            t.data.assign(reinterpret_cast<const int8_t*>(w.ptr),
                reinterpret_cast<const int8_t*>(w.ptr) + w.numel());
            t.scales = SafeTensors::to_f32(st[name + ".scale"]);
            return t;
        };
        auto load_linear = [&](const std::string& name, int rows, int cols) {
            require(name);
            check_shape(name, rows, cols);
            if (st.has(name + ".scale")) return load_int8(name, rows, cols);
            return quantize_per_channel(SafeTensors::to_f32(st[name]), rows, cols);
        };

        init_rope_cache(HD, cfg.rope_theta);

        embed = load_linear("model.embed_tokens.weight", VOCAB, H);
        if (!tied) lm_head = load_linear("lm_head.weight", VOCAB, H);
        finalNorm = load("model.norm.weight");

        layers.resize(NL);
        for (int i = 0; i < NL; ++i) {
            const std::string p = "model.layers." + std::to_string(i) + ".";
            Layer& L = layers[i];
            L.inputNorm = load(p + "input_layernorm.weight");
            L.postNorm = load(p + "post_attention_layernorm.weight");
            if (cfg.use_qk_norm) {
                L.q_norm = load(p + "self_attn.q_norm.weight");
                L.k_norm = load(p + "self_attn.k_norm.weight");
            }
            L.q_w = load_linear(p + "self_attn.q_proj.weight", Q_DIM, H);
            L.k_w = load_linear(p + "self_attn.k_proj.weight", KVD, H);
            L.v_w = load_linear(p + "self_attn.v_proj.weight", KVD, H);
            L.o_w = load_linear(p + "self_attn.o_proj.weight", H, Q_DIM);
            L.gate_w = load_linear(p + "mlp.gate_proj.weight", INTER, H);
            L.up_w = load_linear(p + "mlp.up_proj.weight", INTER, H);
            L.down_w = load_linear(p + "mlp.down_proj.weight", H, INTER);
            if (has_bias) {
                if (st.has(p + "self_attn.q_proj.bias")) L.q_b = SafeTensors::to_f32(st[p + "self_attn.q_proj.bias"]);
                if (st.has(p + "self_attn.k_proj.bias")) L.k_b = SafeTensors::to_f32(st[p + "self_attn.k_proj.bias"]);
                if (st.has(p + "self_attn.v_proj.bias")) L.v_b = SafeTensors::to_f32(st[p + "self_attn.v_proj.bias"]);
            }
        }

        const size_t kv_size = (size_t)NL * kMaxContext * KVD;
        std::fprintf(stderr, "[model] weights loaded, KV cache %.1f MB\n",
            (double)kv_size * 8 / 1e6);
        kv_k_flat.assign(kv_size, 0.0f);
        kv_v_flat.assign(kv_size, 0.0f);

        x.assign(H, 0); xNorm.assign(H, 0);
        q.assign(Q_DIM, 0); k.assign(KVD, 0); v.assign(KVD, 0);
        attnOut.assign(Q_DIM, 0); oProj.assign(H, 0);
        gate.assign(INTER, 0); up.assign(INTER, 0); mlp.assign(H, 0);
        logits.assign(VOCAB, 0);
        scores.assign(kMaxContext, 0);
        scores_buf.assign(kMaxContext, 0);

        Xb.assign((size_t)kMaxPrefill * H, 0);
        Xnb.assign((size_t)kMaxPrefill * H, 0);
        Qb.assign((size_t)kMaxPrefill * Q_DIM, 0);
        Kb.assign((size_t)kMaxPrefill * KVD, 0);
        Vb.assign((size_t)kMaxPrefill * KVD, 0);
        AttnOutb.assign((size_t)kMaxPrefill * Q_DIM, 0);
        OProjb.assign((size_t)kMaxPrefill * H, 0);
        Gateb.assign((size_t)kMaxPrefill * INTER, 0);
        Upb.assign((size_t)kMaxPrefill * INTER, 0);
        Mlb.assign((size_t)kMaxPrefill * H, 0);
    }

    void reset_kv() { kv_len_ = 0; }
    int kv_len() const { return kv_len_; }

    inline float* k_at(int li, int t) {
        return kv_k_flat.data() + ((size_t)li * kMaxContext + t) * KVD;
    }
    inline float* v_at(int li, int t) {
        return kv_v_flat.data() + ((size_t)li * kMaxContext + t) * KVD;
    }

    inline void embed_lookup(int token_id, float* out) const {
        const int8_t* row = embed.data.data() + (size_t)token_id * H;
        const float sc = embed.scales[token_id];
        for (int i = 0; i < H; ++i) out[i] = (float)row[i] * sc;
    }

    // Single-token decode. Caller must ensure kv_len_ < kMaxContext.
    const float* forward(int token_id) {
        if (kv_len_ >= kMaxContext)
            throw std::runtime_error("KV cache full; call reset_kv() first");
        const int pos = kv_len_;
        embed_lookup(token_id, x.data());

        for (int li = 0; li < NL; ++li) {
            Layer& L = layers[li];
            rms_norm(x.data(), L.inputNorm.data(), H, RMS_EPS, xNorm.data());
            matvec_int8_parallel(xNorm.data(), H, L.q_w,
                L.q_b.empty() ? nullptr : L.q_b.data(), q.data(), Q_DIM);
            matvec_int8_parallel(xNorm.data(), H, L.k_w,
                L.k_b.empty() ? nullptr : L.k_b.data(), k.data(), KVD);
            matvec_int8_parallel(xNorm.data(), H, L.v_w,
                L.v_b.empty() ? nullptr : L.v_b.data(), v.data(), KVD);

            if (cfg.use_qk_norm) {
                for (int h = 0; h < NH; ++h)
                    rms_norm(q.data() + h * HD, L.q_norm.data(), HD, RMS_EPS, q.data() + h * HD);
                for (int h = 0; h < NKV; ++h)
                    rms_norm(k.data() + h * HD, L.k_norm.data(), HD, RMS_EPS, k.data() + h * HD);
            }
            for (int h = 0; h < NH; ++h) apply_rope(q.data(), h * HD, HD, pos);
            for (int h = 0; h < NKV; ++h) apply_rope(k.data(), h * HD, HD, pos);

            std::memcpy(k_at(li, pos), k.data(), KVD * sizeof(float));
            std::memcpy(v_at(li, pos), v.data(), KVD * sizeof(float));

            const int seq_len = pos + 1;
            const float scale = 1.0f / std::sqrt((float)HD);
            const float* kbase = kv_k_flat.data() + (size_t)li * kMaxContext * KVD;
            const float* vbase = kv_v_flat.data() + (size_t)li * kMaxContext * KVD;

            for (int h = 0; h < NH; ++h) {
                const int kv_head = h / GROUPS;
                const int q_off = h * HD;
                const int kv_off = kv_head * HD;
                for (int t = 0; t < seq_len; ++t) {
                    const float* kt = kbase + (size_t)t * KVD + kv_off;
                    float s = 0.0f;
                    for (int d = 0; d < HD; ++d) s += q[q_off + d] * kt[d];
                    scores[t] = s * scale;
                }
                softmax_inplace(scores.data(), seq_len);
                for (int d = 0; d < HD; ++d) {
                    float s = 0.0f;
                    for (int t = 0; t < seq_len; ++t)
                        s += scores[t] * vbase[(size_t)t * KVD + kv_off + d];
                    attnOut[q_off + d] = s;
                }
            }

            matvec_int8_parallel(attnOut.data(), Q_DIM, L.o_w, nullptr, oProj.data(), H);
            for (int i = 0; i < H; ++i) x[i] += oProj[i];

            rms_norm(x.data(), L.postNorm.data(), H, RMS_EPS, xNorm.data());
            matvec_int8_parallel(xNorm.data(), H, L.gate_w, nullptr, gate.data(), INTER);
            matvec_int8_parallel(xNorm.data(), H, L.up_w, nullptr, up.data(), INTER);
            silu_inplace(gate.data(), INTER);
            for (int i = 0; i < INTER; ++i) gate[i] *= up[i];
            matvec_int8_parallel(gate.data(), INTER, L.down_w, nullptr, mlp.data(), H);
            for (int i = 0; i < H; ++i) x[i] += mlp[i];
        }

        rms_norm(x.data(), finalNorm.data(), H, RMS_EPS, xNorm.data());
        const Int8Tensor& head = cfg.tie_word_embeddings ? embed : lm_head;
        matvec_int8_parallel(xNorm.data(), H, head, nullptr, logits.data(), VOCAB);
        kv_len_++;
        return logits.data();
    }

    // Batched prefill. Caller must ensure kv_len_ + N <= kMaxContext.
    const float* forward_batch(const std::vector<int>& tokens) {
        const int N = (int)tokens.size();
        if (N == 0) return logits.data();
        if (N > kMaxPrefill)
            throw std::runtime_error("forward_batch: N exceeds kMaxPrefill");
        if (kv_len_ + N > kMaxContext)
            throw std::runtime_error("forward_batch: would exceed context; reset first");

        const int past_len = kv_len_;
        for (int t = 0; t < N; ++t)
            embed_lookup(tokens[t], Xb.data() + (size_t)t * H);

        for (int li = 0; li < NL; ++li) {
            Layer& L = layers[li];
            rms_norm_batch(Xb.data(), N, H, L.inputNorm.data(), RMS_EPS, Xnb.data());
            matvec_int8_batch(Xnb.data(), N, H, L.q_w,
                L.q_b.empty() ? nullptr : L.q_b.data(), Qb.data(), Q_DIM);
            matvec_int8_batch(Xnb.data(), N, H, L.k_w,
                L.k_b.empty() ? nullptr : L.k_b.data(), Kb.data(), KVD);
            matvec_int8_batch(Xnb.data(), N, H, L.v_w,
                L.v_b.empty() ? nullptr : L.v_b.data(), Vb.data(), KVD);

            if (cfg.use_qk_norm) {
                for (int t = 0; t < N; ++t) {
                    float* qrow = Qb.data() + (size_t)t * Q_DIM;
                    for (int h = 0; h < NH; ++h)
                        rms_norm(qrow + h * HD, L.q_norm.data(), HD, RMS_EPS, qrow + h * HD);
                    float* krow = Kb.data() + (size_t)t * KVD;
                    for (int h = 0; h < NKV; ++h)
                        rms_norm(krow + h * HD, L.k_norm.data(), HD, RMS_EPS, krow + h * HD);
                }
            }
            for (int t = 0; t < N; ++t) {
                const int pos = past_len + t;
                for (int h = 0; h < NH; ++h)
                    apply_rope(Qb.data(), t * Q_DIM + h * HD, HD, pos);
                for (int h = 0; h < NKV; ++h)
                    apply_rope(Kb.data(), t * KVD + h * HD, HD, pos);
            }
            for (int t = 0; t < N; ++t) {
                std::memcpy(k_at(li, past_len + t), Kb.data() + (size_t)t * KVD, KVD * sizeof(float));
                std::memcpy(v_at(li, past_len + t), Vb.data() + (size_t)t * KVD, KVD * sizeof(float));
            }

            const float scale = 1.0f / std::sqrt((float)HD);
            float* sc = scores_buf.data();
            const float* kbase = kv_k_flat.data() + (size_t)li * kMaxContext * KVD;
            const float* vbase = kv_v_flat.data() + (size_t)li * kMaxContext * KVD;

            for (int h = 0; h < NH; ++h) {
                const int kv_head = h / GROUPS;
                const int q_off = h * HD;
                const int kv_off = kv_head * HD;
                for (int t1 = 0; t1 < N; ++t1) {
                    const float* qrow = Qb.data() + (size_t)t1 * Q_DIM + q_off;
                    const int cnt = past_len + t1 + 1;
                    for (int t2 = 0; t2 < cnt; ++t2) {
                        const float* krow = kbase + (size_t)t2 * KVD + kv_off;
                        __m128 sum = _mm_setzero_ps();
                        int d = 0;
                        for (; d + 4 <= HD; d += 4) {
                            const __m128 qv = _mm_loadu_ps(qrow + d);
                            const __m128 kv = _mm_loadu_ps(krow + d);
                            sum = _mm_add_ps(sum, _mm_mul_ps(qv, kv));
                        }
                        float s = hsum_sse(sum);
                        for (; d < HD; ++d) s += qrow[d] * krow[d];
                        sc[t2] = s * scale;
                    }
                    softmax_inplace(sc, cnt);
                    float* orow = AttnOutb.data() + (size_t)t1 * Q_DIM + q_off;
                    for (int d = 0; d < HD; ++d) orow[d] = 0.0f;
                    for (int t2 = 0; t2 < cnt; ++t2) {
                        const float w = sc[t2];
                        const float* vrow = vbase + (size_t)t2 * KVD + kv_off;
                        for (int d = 0; d < HD; ++d) orow[d] += w * vrow[d];
                    }
                }
            }

            matvec_int8_batch(AttnOutb.data(), N, Q_DIM, L.o_w, nullptr, OProjb.data(), H);
            for (int i = 0; i < N * H; ++i) Xb[i] += OProjb[i];
            rms_norm_batch(Xb.data(), N, H, L.postNorm.data(), RMS_EPS, Xnb.data());
            matvec_int8_batch(Xnb.data(), N, H, L.gate_w, nullptr, Gateb.data(), INTER);
            matvec_int8_batch(Xnb.data(), N, H, L.up_w, nullptr, Upb.data(), INTER);
            silu_inplace(Gateb.data(), N * INTER);
            for (int i = 0; i < N * INTER; ++i) Gateb[i] *= Upb[i];
            matvec_int8_batch(Gateb.data(), N, INTER, L.down_w, nullptr, Mlb.data(), H);
            for (int i = 0; i < N * H; ++i) Xb[i] += Mlb[i];
        }

        rms_norm_batch(Xb.data(), N, H, finalNorm.data(), RMS_EPS, Xnb.data());
        const float* last_xn = Xnb.data() + (size_t)(N - 1) * H;
        const Int8Tensor& head = cfg.tie_word_embeddings ? embed : lm_head;
        matvec_int8_parallel(last_xn, H, head, nullptr, logits.data(), VOCAB);
        kv_len_ += N;
        return logits.data();
    }
};