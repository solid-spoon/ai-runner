// cpp/model.h Ч Qwen2 forward pass with flat KV cache.
#pragma once

#include "st.h"
#include "matmul.h"
#include "quant.h"

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

inline constexpr int kMaxPrefill = 1024;
// Bump this if you need longer context. Memory cost is
//   2 * NL * kMaxContext * KVD * 4 bytes
//   = 2 * 24 * 8192 * 128 * 4 = ~201 MB for kMaxContext=8192.
inline constexpr int kMaxContext = 8192;

class Qwen2Model {
public:
    Qwen2Config cfg;

    int H, NH, NKV, NL, HD, KVD, INTER, VOCAB, GROUPS;
    float RMS_EPS, ROPE_THETA;

    Int8Tensor embed;
    std::vector<float> finalNorm;

    struct Layer {
        std::vector<float> inputNorm;
        std::vector<float> postNorm;

        Int8Tensor q_w, k_w, v_w, o_w;
        std::vector<float> q_b, k_b, v_b;

        Int8Tensor gate_w, up_w, down_w;
    };
    std::vector<Layer> layers;

    // Flat KV cache.
    //   Layout: kv_k_flat[li * (kMaxContext * KVD) + t * KVD + c]
    //   li = layer in [0, NL), t = position, c = channel in [0, KVD).
    //   All layers share a single kv_len_ Ч they always have the same length.
    std::vector<float> kv_k_flat;
    std::vector<float> kv_v_flat;
    int kv_len_ = 0;

    // Single-token scratch.
    std::vector<float> x, xNorm;
    std::vector<float> q, k, v, attnOut, oProj;
    std::vector<float> gate, up, mlp;
    std::vector<float> logits;
    std::vector<float> scores;

    // Batched (prefill) scratch.
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
        std::fprintf(stderr,
            "[model] flat KV cache: %d layers x %d positions x %d ch = %.1f MB\n",
            NL, kMaxContext, KVD,
            (2.0 * NL * kMaxContext * KVD * 4) / 1e6);

        auto load = [&](const std::string& name) {
            return SafeTensors::to_f32(st[name]);
        };

        auto load_int8 = [&](const std::string& name,
            int rows, int cols) -> Int8Tensor {
                Int8Tensor t;
                t.rows = rows;
                t.cols = cols;

                const TensorInfo& w = st[name];
                const TensorInfo& s = st[name + ".scale"];

                t.data.assign(
                    reinterpret_cast<const int8_t*>(w.ptr),
                    reinterpret_cast<const int8_t*>(w.ptr) + w.numel());

                t.scales = SafeTensors::to_f32(s);
                return t;
        };

        auto load_linear = [&](const std::string& name,
            int rows, int cols) -> Int8Tensor {
                if (st.has(name + ".scale")) {
                    return load_int8(name, rows, cols);
                }
                auto fp32 = SafeTensors::to_f32(st[name]);
                return quantize_per_channel(fp32, rows, cols);
        };

        const bool prequant =
            st.has("model.layers.0.self_attn.q_proj.weight.scale");
        std::fprintf(stderr, "[model] linear weights: %s\n",
            prequant ? "pre-quantized INT8 (fast load)"
            : "FP32/BF16 (quantizing on the fly)");

        // RoPE cos/sin cache.
        init_rope_cache(HD, ROPE_THETA);

        embed = load_linear("model.embed_tokens.weight", VOCAB, H);
        finalNorm = load("model.norm.weight");

        layers.resize(NL);
        for (int i = 0; i < NL; ++i) {
            const std::string p = "model.layers." + std::to_string(i) + ".";
            Layer& L = layers[i];

            L.inputNorm = load(p + "input_layernorm.weight");
            L.postNorm = load(p + "post_attention_layernorm.weight");

            L.q_w = load_linear(p + "self_attn.q_proj.weight", H, H);
            L.k_w = load_linear(p + "self_attn.k_proj.weight", KVD, H);
            L.v_w = load_linear(p + "self_attn.v_proj.weight", KVD, H);
            L.o_w = load_linear(p + "self_attn.o_proj.weight", H, H);

            L.gate_w = load_linear(p + "mlp.gate_proj.weight", INTER, H);
            L.up_w = load_linear(p + "mlp.up_proj.weight", INTER, H);
            L.down_w = load_linear(p + "mlp.down_proj.weight", H, INTER);

            L.q_b = load(p + "self_attn.q_proj.bias");
            L.k_b = load(p + "self_attn.k_proj.bias");
            L.v_b = load(p + "self_attn.v_proj.bias");
        }
        std::fprintf(stderr, "[model] weights loaded\n");

        // Flat KV cache Ч one allocation per matrix.
        const size_t kv_size =
            static_cast<size_t>(NL) * kMaxContext * KVD;
        kv_k_flat.assign(kv_size, 0.0f);
        kv_v_flat.assign(kv_size, 0.0f);
        kv_len_ = 0;

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
    }

    // O(1) Ч никакого обхода 24 слоЄв.
    void reset_kv() { kv_len_ = 0; }

    int kv_len() const { return kv_len_; }

    // Pointer to the start of (layer li, position t) inside the flat K cache.
    inline float* k_at(int li, int t) {
        return kv_k_flat.data()
            + (static_cast<size_t>(li) * kMaxContext + t) * KVD;
    }
    inline const float* k_at(int li, int t) const {
        return kv_k_flat.data()
            + (static_cast<size_t>(li) * kMaxContext + t) * KVD;
    }
    inline float* v_at(int li, int t) {
        return kv_v_flat.data()
            + (static_cast<size_t>(li) * kMaxContext + t) * KVD;
    }
    inline const float* v_at(int li, int t) const {
        return kv_v_flat.data()
            + (static_cast<size_t>(li) * kMaxContext + t) * KVD;
    }

    inline void embed_lookup(int token_id, float* __restrict out) const {
        const int8_t* row = embed.data.data()
            + static_cast<size_t>(token_id) * H;
        const float sc = embed.scales[token_id];
        for (int i = 0; i < H; ++i) out[i] = static_cast<float>(row[i]) * sc;
    }

    // ??? Single-token forward ?????????????????????????????????
    const float* forward(int token_id, int position) {
        // Guard against overflow.
        if (kv_len_ >= kMaxContext) {
            std::fprintf(stderr, "[warn] context full, resetting KV cache\n");
            reset_kv();
            position = 0;
        }

        const int pos = kv_len_;   // position we're writing this token at

        embed_lookup(token_id, x.data());

        for (int li = 0; li < NL; ++li) {
            Layer& L = layers[li];

            rms_norm(x.data(), L.inputNorm.data(), H, RMS_EPS, xNorm.data());

            matvec_int8_parallel(xNorm.data(), H, L.q_w, L.q_b.data(), q.data(), H);
            matvec_int8_parallel(xNorm.data(), H, L.k_w, L.k_b.data(), k.data(), KVD);
            matvec_int8_parallel(xNorm.data(), H, L.v_w, L.v_b.data(), v.data(), KVD);

            for (int h = 0; h < NH; ++h)
                apply_rope(q.data(), h * HD, HD, pos, ROPE_THETA);
            for (int h = 0; h < NKV; ++h)
                apply_rope(k.data(), h * HD, HD, pos, ROPE_THETA);

            // Write this position into the flat cache.
            std::memcpy(k_at(li, pos), k.data(), KVD * sizeof(float));
            std::memcpy(v_at(li, pos), v.data(), KVD * sizeof(float));

            const int seq_len = pos + 1;

            const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
            const float* kbase = kv_k_flat.data()
                + static_cast<size_t>(li) * kMaxContext * KVD;
            const float* vbase = kv_v_flat.data()
                + static_cast<size_t>(li) * kMaxContext * KVD;

            for (int h = 0; h < NH; ++h) {
                const int kv_head = h / GROUPS;
                const int q_off = h * HD;
                const int kv_off = kv_head * HD;

                for (int t = 0; t < seq_len; ++t) {
                    const float* kt = kbase + static_cast<size_t>(t) * KVD + kv_off;
                    float s = 0.0f;
                    for (int d = 0; d < HD; ++d) s += q[q_off + d] * kt[d];
                    scores[t] = s * scale;
                }
                softmax_inplace(scores.data(), seq_len);

                for (int d = 0; d < HD; ++d) {
                    float s = 0.0f;
                    for (int t = 0; t < seq_len; ++t)
                        s += scores[t] * vbase[static_cast<size_t>(t) * KVD + kv_off + d];
                    attnOut[q_off + d] = s;
                }
            }

            matvec_int8_parallel(attnOut.data(), H, L.o_w, nullptr,
                oProj.data(), H);
            for (int i = 0; i < H; ++i) x[i] += oProj[i];

            rms_norm(x.data(), L.postNorm.data(), H, RMS_EPS, xNorm.data());

            matvec_int8_parallel(xNorm.data(), H, L.gate_w, nullptr,
                gate.data(), INTER);
            matvec_int8_parallel(xNorm.data(), H, L.up_w, nullptr,
                up.data(), INTER);
            silu_inplace(gate.data(), INTER);
            for (int i = 0; i < INTER; ++i) gate[i] *= up[i];
            matvec_int8_parallel(gate.data(), INTER, L.down_w, nullptr,
                mlp.data(), H);

            for (int i = 0; i < H; ++i) x[i] += mlp[i];
        }

        rms_norm(x.data(), finalNorm.data(), H, RMS_EPS, xNorm.data());
        matvec_int8_parallel(xNorm.data(), H, embed, nullptr,
            logits.data(), VOCAB);

        kv_len_++;
        return logits.data();
    }

    // ??? Batched forward (prefill) ????????????????????????????
    const float* forward_batch(const std::vector<int>& tokens) {
        int N = static_cast<int>(tokens.size());
        if (N == 0) return logits.data();

        if (N > kMaxPrefill) {
            std::vector<int> tail(tokens.end() - kMaxPrefill, tokens.end());
            return forward_batch(tail);
        }

        // Guard against context overflow.
        if (kv_len_ + N > kMaxContext) {
            std::fprintf(stderr,
                "[warn] context would overflow (%d + %d > %d), resetting\n",
                kv_len_, N, kMaxContext);
            reset_kv();
            if (N > kMaxContext) {
                std::vector<int> tail(tokens.end() - kMaxContext, tokens.end());
                return forward_batch(tail);
            }
        }

        const int past_len = kv_len_;

        // Embedding lookup.
        for (int t = 0; t < N; ++t) {
            embed_lookup(tokens[t], Xb.data() + static_cast<size_t>(t) * H);
        }

        for (int li = 0; li < NL; ++li) {
            Layer& L = layers[li];

            rms_norm_batch(Xb.data(), N, H, L.inputNorm.data(),
                RMS_EPS, Xnb.data());

            matvec_int8_batch(Xnb.data(), N, H, L.q_w, L.q_b.data(),
                Qb.data(), H);
            matvec_int8_batch(Xnb.data(), N, H, L.k_w, L.k_b.data(),
                Kb.data(), KVD);
            matvec_int8_batch(Xnb.data(), N, H, L.v_w, L.v_b.data(),
                Vb.data(), KVD);

            for (int t = 0; t < N; ++t) {
                const int pos = past_len + t;
                for (int h = 0; h < NH; ++h)
                    apply_rope(Qb.data(), t * H + h * HD, HD, pos, ROPE_THETA);
                for (int h = 0; h < NKV; ++h)
                    apply_rope(Kb.data(), t * KVD + h * HD, HD, pos, ROPE_THETA);
            }

            // Write N new positions into the flat cache.
            for (int t = 0; t < N; ++t) {
                std::memcpy(k_at(li, past_len + t),
                    Kb.data() + static_cast<size_t>(t) * KVD,
                    KVD * sizeof(float));
                std::memcpy(v_at(li, past_len + t),
                    Vb.data() + static_cast<size_t>(t) * KVD,
                    KVD * sizeof(float));
            }

            const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
            float* sc = scores_buf.data();
            const float* kbase = kv_k_flat.data()
                + static_cast<size_t>(li) * kMaxContext * KVD;
            const float* vbase = kv_v_flat.data()
                + static_cast<size_t>(li) * kMaxContext * KVD;

            for (int h = 0; h < NH; ++h) {
                const int kv_head = h / GROUPS;
                const int q_off = h * HD;
                const int kv_off = kv_head * HD;

                for (int t1 = 0; t1 < N; ++t1) {
                    const float* qrow =
                        Qb.data() + static_cast<size_t>(t1) * H + q_off;
                    const int cnt = past_len + t1 + 1;

                    for (int t2 = 0; t2 < cnt; ++t2) {
                        const float* krow = kbase + static_cast<size_t>(t2) * KVD + kv_off;
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

                    float* orow =
                        AttnOutb.data() + static_cast<size_t>(t1) * H + q_off;
                    for (int d = 0; d < HD; ++d) orow[d] = 0.0f;
                    for (int t2 = 0; t2 < cnt; ++t2) {
                        const float w = sc[t2];
                        const float* vrow = vbase + static_cast<size_t>(t2) * KVD + kv_off;
                        for (int d = 0; d < HD; ++d) orow[d] += w * vrow[d];
                    }
                }
            }

            matvec_int8_batch(AttnOutb.data(), N, H, L.o_w, nullptr,
                OProjb.data(), H);
            for (int i = 0; i < N * H; ++i) Xb[i] += OProjb[i];

            rms_norm_batch(Xb.data(), N, H, L.postNorm.data(),
                RMS_EPS, Xnb.data());

            matvec_int8_batch(Xnb.data(), N, H, L.gate_w, nullptr,
                Gateb.data(), INTER);
            matvec_int8_batch(Xnb.data(), N, H, L.up_w, nullptr,
                Upb.data(), INTER);
            silu_inplace(Gateb.data(), N * INTER);
            for (int i = 0; i < N * INTER; ++i) Gateb[i] *= Upb[i];
            matvec_int8_batch(Gateb.data(), N, INTER, L.down_w, nullptr,
                Mlb.data(), H);

            for (int i = 0; i < N * H; ++i) Xb[i] += Mlb[i];
        }

        rms_norm_batch(Xb.data(), N, H, finalNorm.data(), RMS_EPS, Xnb.data());

        const float* last_xn = Xnb.data() + static_cast<size_t>(N - 1) * H;
        matvec_int8_parallel(last_xn, H, embed, nullptr,
            logits.data(), VOCAB);

        kv_len_ += N;
        return logits.data();
    }
};