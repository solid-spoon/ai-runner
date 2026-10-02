// cpp/model.h — Qwen2 forward pass (FP32 weights, single-token + batched prefill).
#pragma once

#include "st.h"
#include "matmul.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

struct Qwen2Config {
    int   hidden_size = 896;
    int   num_attention_heads = 14;
    int   num_key_value_heads = 2;
    int   num_hidden_layers = 24;
    int   intermediate_size = 4864;
    int   vocab_size = 151936;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    int   max_position_embeddings = 32768;
};

// Scratch buffers are preallocated to these bounds.
inline constexpr int kMaxPrefill = 1024;
inline constexpr int kMaxContext = 8192;

class Qwen2Model {
public:
    Qwen2Config cfg;

    // Derived dimensions, cached for convenience.
    int H, NH, NKV, NL, HD, KVD, INTER, VOCAB, GROUPS;
    float RMS_EPS, ROPE_THETA;

    // Global tensors.
    std::vector<float> embed;
    std::vector<float> finalNorm;

    struct Layer {
        std::vector<float> inputNorm;
        std::vector<float> q_w, q_b;
        std::vector<float> k_w, k_b;
        std::vector<float> v_w, v_b;
        std::vector<float> o_w;
        std::vector<float> postNorm;
        std::vector<float> gate_w, up_w, down_w;
    };
    std::vector<Layer> layers;

    // KV cache: [layer][seq_pos][channel].
    std::vector<std::vector<std::vector<float>>> kv_k;
    std::vector<std::vector<std::vector<float>>> kv_v;

    // Single-token scratch buffers.
    std::vector<float> x, xNorm;
    std::vector<float> q, k, v, attnOut, oProj;
    std::vector<float> gate, up, mlp;
    std::vector<float> logits;
    std::vector<float> scores;

    // Batched (prefill) scratch buffers.
    std::vector<float> Xb, Xnb, Qb, Kb, Vb, AttnOutb, OProjb;
    std::vector<float> Gateb, Upb, Mlb;
    std::vector<float> scores_buf;

    explicit Qwen2Model(const SafeTensors& st, const Qwen2Config& c = {})
        : cfg(c) {
        H = cfg.hidden_size;
        NH = cfg.num_attention_heads;
        NKV = cfg.num_key_value_heads;
        NL = cfg.num_hidden_layers;
        HD = H / NH;
        KVD = NKV * HD;
        INTER = cfg.intermediate_size;
        VOCAB = cfg.vocab_size;
        RMS_EPS = cfg.rms_norm_eps;
        ROPE_THETA = cfg.rope_theta;
        GROUPS = NH / NKV;

        std::fprintf(stderr,
            "[model] H=%d NH=%d NKV=%d NL=%d HD=%d KVD=%d INTER=%d VOCAB=%d\n",
            H, NH, NKV, NL, HD, KVD, INTER, VOCAB);

        auto load = [&](const std::string& name) {
            return SafeTensors::to_f32(st[name]);
        };

        embed = load("model.embed_tokens.weight");
        finalNorm = load("model.norm.weight");

        layers.resize(NL);
        for (int i = 0; i < NL; ++i) {
            const std::string p = "model.layers." + std::to_string(i) + ".";
            Layer& L = layers[i];
            L.inputNorm = load(p + "input_layernorm.weight");
            L.q_w = load(p + "self_attn.q_proj.weight");
            L.k_w = load(p + "self_attn.k_proj.weight");
            L.v_w = load(p + "self_attn.v_proj.weight");
            L.o_w = load(p + "self_attn.o_proj.weight");
            L.q_b = load(p + "self_attn.q_proj.bias");
            L.k_b = load(p + "self_attn.k_proj.bias");
            L.v_b = load(p + "self_attn.v_proj.bias");
            L.postNorm = load(p + "post_attention_layernorm.weight");
            L.gate_w = load(p + "mlp.gate_proj.weight");
            L.up_w = load(p + "mlp.up_proj.weight");
            L.down_w = load(p + "mlp.down_proj.weight");
        }

        // Single-token buffers.
        x.assign(H, 0);          xNorm.assign(H, 0);
        q.assign(H, 0);          k.assign(KVD, 0);    v.assign(KVD, 0);
        attnOut.assign(H, 0);    oProj.assign(H, 0);
        gate.assign(INTER, 0);   up.assign(INTER, 0); mlp.assign(H, 0);
        logits.assign(VOCAB, 0);
        scores.assign(kMaxContext, 0);

        // Batched buffers.
        Xb.assign(static_cast<size_t>(kMaxPrefill) * H, 0);
        Xnb.assign(static_cast<size_t>(kMaxPrefill) * H, 0);
        Qb.assign(static_cast<size_t>(kMaxPrefill) * H, 0);
        Kb.assign(static_cast<size_t>(kMaxPrefill) * KVD, 0);
        Vb.assign(static_cast<size_t>(kMaxPrefill) * KVD, 0);
        AttnOutb.assign(static_cast<size_t>(kMaxPrefill) * H, 0);
        OProjb.assign(static_cast<size_t>(kMaxPrefill) * H, 0);
        Gateb.assign(static_cast<size_t>(kMaxPrefill) * INTER, 0);
        Upb.assign(static_cast<size_t>(kMaxPrefill) * INTER, 0);
        Mlb.assign(static_cast<size_t>(kMaxPrefill) * H, 0);
        scores_buf.assign(kMaxContext, 0);

        kv_k.resize(NL);
        kv_v.resize(NL);
        for (int i = 0; i < NL; ++i) {
            kv_k[i].reserve(kMaxContext);
            kv_v[i].reserve(kMaxContext);
        }
    }

    void reset_kv() {
        for (int i = 0; i < NL; ++i) {
            kv_k[i].clear();
            kv_v[i].clear();
        }
    }

    int kv_len() const { return static_cast<int>(kv_k[0].size()); }

    // ??? Single-token forward pass (for generation) ????????????????????
    // Returns a pointer to the logits buffer.
    const float* forward(int token_id, int position) {
        const float* emb = embed.data() + static_cast<size_t>(token_id) * H;
        std::memcpy(x.data(), emb, H * sizeof(float));

        for (int li = 0; li < NL; ++li) {
            Layer& L = layers[li];

            // Pre-attention norm.
            rms_norm(x.data(), L.inputNorm.data(), H, RMS_EPS, xNorm.data());

            // QKV projections.
            matvec_parallel(xNorm.data(), H, L.q_w.data(), L.q_b.data(), q.data(), H);
            matvec_parallel(xNorm.data(), H, L.k_w.data(), L.k_b.data(), k.data(), KVD);
            matvec_parallel(xNorm.data(), H, L.v_w.data(), L.v_b.data(), v.data(), KVD);

            // Rotary embeddings.
            for (int h = 0; h < NH; ++h)
                apply_rope(q.data(), h * HD, HD, position, ROPE_THETA);
            for (int h = 0; h < NKV; ++h)
                apply_rope(k.data(), h * HD, HD, position, ROPE_THETA);

            kv_k[li].emplace_back(k);
            kv_v[li].emplace_back(v);
            const int seq_len = static_cast<int>(kv_k[li].size());

            // Attention with GQA.
            const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
            for (int h = 0; h < NH; ++h) {
                const int kv_head = h / GROUPS;
                const int q_off = h * HD;
                const int kv_off = kv_head * HD;

                for (int t = 0; t < seq_len; ++t) {
                    const float* kt = kv_k[li][t].data() + kv_off;
                    float s = 0.0f;
                    for (int d = 0; d < HD; ++d) s += q[q_off + d] * kt[d];
                    scores[t] = s * scale;
                }
                softmax_inplace(scores.data(), seq_len);

                for (int d = 0; d < HD; ++d) {
                    float s = 0.0f;
                    for (int t = 0; t < seq_len; ++t)
                        s += scores[t] * kv_v[li][t][kv_off + d];
                    attnOut[q_off + d] = s;
                }
            }

            // Output projection + residual.
            matvec_parallel(attnOut.data(), H, L.o_w.data(), nullptr,
                oProj.data(), H);
            for (int i = 0; i < H; ++i) x[i] += oProj[i];

            // Post-attention norm.
            rms_norm(x.data(), L.postNorm.data(), H, RMS_EPS, xNorm.data());

            // SwiGLU MLP.
            matvec_parallel(xNorm.data(), H, L.gate_w.data(), nullptr,
                gate.data(), INTER);
            matvec_parallel(xNorm.data(), H, L.up_w.data(), nullptr,
                up.data(), INTER);
            silu_inplace(gate.data(), INTER);
            for (int i = 0; i < INTER; ++i) gate[i] *= up[i];
            matvec_parallel(gate.data(), INTER, L.down_w.data(), nullptr,
                mlp.data(), H);

            for (int i = 0; i < H; ++i) x[i] += mlp[i];
        }

        // Final norm + LM head.
        rms_norm(x.data(), finalNorm.data(), H, RMS_EPS, xNorm.data());
        matvec_parallel(xNorm.data(), H, embed.data(), nullptr,
            logits.data(), VOCAB);
        return logits.data();
    }

    // ??? Batched forward pass (for prefill) ????????????????????????????
    // Returns logits for the last token only.
    const float* forward_batch(const std::vector<int>& tokens) {
        const int N = static_cast<int>(tokens.size());
        if (N == 0) return logits.data();

        // Truncate overly long prompts to the last kMaxPrefill tokens.
        if (N > kMaxPrefill) {
            std::vector<int> tail(tokens.end() - kMaxPrefill, tokens.end());
            return forward_batch(tail);
        }

        const int past_len = kv_len();

        // Embedding lookup.
        for (int t = 0; t < N; ++t) {
            const float* emb =
                embed.data() + static_cast<size_t>(tokens[t]) * H;
            std::memcpy(Xb.data() + static_cast<size_t>(t) * H,
                emb, H * sizeof(float));
        }

        for (int li = 0; li < NL; ++li) {
            Layer& L = layers[li];

            // Pre-attention norm.
            rms_norm_batch(Xb.data(), N, H, L.inputNorm.data(),
                RMS_EPS, Xnb.data());

            // QKV projections.
            matvec_batch(Xnb.data(), N, H, L.q_w.data(), L.q_b.data(),
                Qb.data(), H);
            matvec_batch(Xnb.data(), N, H, L.k_w.data(), L.k_b.data(),
                Kb.data(), KVD);
            matvec_batch(Xnb.data(), N, H, L.v_w.data(), L.v_b.data(),
                Vb.data(), KVD);

            // Rotary embeddings.
            for (int t = 0; t < N; ++t) {
                const int pos = past_len + t;
                for (int h = 0; h < NH; ++h)
                    apply_rope(Qb.data(), t * H + h * HD, HD, pos, ROPE_THETA);
                for (int h = 0; h < NKV; ++h)
                    apply_rope(Kb.data(), t * KVD + h * HD, HD, pos, ROPE_THETA);
            }

            // Append to the KV cache.
            for (int t = 0; t < N; ++t) {
                kv_k[li].emplace_back(
                    Kb.data() + static_cast<size_t>(t) * KVD,
                    Kb.data() + static_cast<size_t>(t + 1) * KVD);
                kv_v[li].emplace_back(
                    Vb.data() + static_cast<size_t>(t) * KVD,
                    Vb.data() + static_cast<size_t>(t + 1) * KVD);
            }

            // Causal attention.
            const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
            float* sc = scores_buf.data();

            for (int h = 0; h < NH; ++h) {
                const int kv_head = h / GROUPS;
                const int q_off = h * HD;
                const int kv_off = kv_head * HD;

                for (int t1 = 0; t1 < N; ++t1) {
                    const float* qrow =
                        Qb.data() + static_cast<size_t>(t1) * H + q_off;
                    const int cnt = past_len + t1 + 1;

                    // Dot products with all attended keys.
                    for (int t2 = 0; t2 < cnt; ++t2) {
                        const float* krow = kv_k[li][t2].data() + kv_off;
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

                    // Weighted sum over values.
                    float* orow =
                        AttnOutb.data() + static_cast<size_t>(t1) * H + q_off;
                    for (int d = 0; d < HD; ++d) orow[d] = 0.0f;
                    for (int t2 = 0; t2 < cnt; ++t2) {
                        const float w = sc[t2];
                        const float* vrow = kv_v[li][t2].data() + kv_off;
                        for (int d = 0; d < HD; ++d) orow[d] += w * vrow[d];
                    }
                }
            }

            // Output projection + residual.
            matvec_batch(AttnOutb.data(), N, H, L.o_w.data(), nullptr,
                OProjb.data(), H);
            for (int i = 0; i < N * H; ++i) Xb[i] += OProjb[i];

            // Post-attention norm.
            rms_norm_batch(Xb.data(), N, H, L.postNorm.data(),
                RMS_EPS, Xnb.data());

            // SwiGLU MLP.
            matvec_batch(Xnb.data(), N, H, L.gate_w.data(), nullptr,
                Gateb.data(), INTER);
            matvec_batch(Xnb.data(), N, H, L.up_w.data(), nullptr,
                Upb.data(), INTER);
            silu_inplace(Gateb.data(), N * INTER);
            for (int i = 0; i < N * INTER; ++i) Gateb[i] *= Upb[i];
            matvec_batch(Gateb.data(), N, INTER, L.down_w.data(), nullptr,
                Mlb.data(), H);

            for (int i = 0; i < N * H; ++i) Xb[i] += Mlb[i];
        }

        // Final norm + LM head on the last token only.
        rms_norm_batch(Xb.data(), N, H, finalNorm.data(), RMS_EPS, Xnb.data());

        const float* last_xn =
            Xnb.data() + static_cast<size_t>(N - 1) * H;
        matvec_parallel(last_xn, H, embed.data(), nullptr,
            logits.data(), VOCAB);

        return logits.data();
    }
};