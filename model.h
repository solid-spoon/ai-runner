// cpp/model.h — Qwen2 forward pass (FP32 weights)
#pragma once
#include "st.h"
#include "matmul.h"
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <cstdio>

struct Qwen2Config {
    int hidden_size = 896;
    int num_attention_heads = 14;
    int num_key_value_heads = 2;
    int num_hidden_layers = 24;
    int intermediate_size = 4864;
    int vocab_size = 151936;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    int max_position_embeddings = 32768;
};

class Qwen2Model {
public:
    Qwen2Config cfg;
    int H, NH, NKV, NL, HD, KVD, INTER, VOCAB, GROUPS;
    float RMS_EPS, ROPE_THETA;

    std::vector<float> embed;      // [VOCAB, H]
    std::vector<float> finalNorm;  // [H]

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

    // KV cache: [layer][pos] -> KVD floats
    std::vector<std::vector<std::vector<float>>> kv_k;
    std::vector<std::vector<std::vector<float>>> kv_v;

    // Scratch
    std::vector<float> x, xNorm;
    std::vector<float> q, k, v, attnOut, oProj;
    std::vector<float> gate, up, mlp;
    std::vector<float> logits;
    std::vector<float> scores;

    Qwen2Model(const SafeTensors& st, const Qwen2Config& c = {}) : cfg(c) {
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
        for (int i = 0; i < NL; i++) {
            std::string p = "model.layers." + std::to_string(i) + ".";
            auto& L = layers[i];
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

        x.assign(H, 0);       xNorm.assign(H, 0);
        q.assign(H, 0);       k.assign(KVD, 0);     v.assign(KVD, 0);
        attnOut.assign(H, 0); oProj.assign(H, 0);
        gate.assign(INTER, 0); up.assign(INTER, 0); mlp.assign(H, 0);
        logits.assign(VOCAB, 0);
        scores.assign(cfg.max_position_embeddings, 0);

        kv_k.resize(NL);
        kv_v.resize(NL);
    }

    void reset_kv() {
        for (int i = 0; i < NL; i++) { kv_k[i].clear(); kv_v[i].clear(); }
    }

    // Один forward pass. Возвращает указатель на logits.
    const float* forward(int tokenId, int position) {
        // Embedding
        const float* emb = embed.data() + (size_t)tokenId * H;
        std::memcpy(x.data(), emb, H * sizeof(float));

        for (int li = 0; li < NL; li++) {
            auto& L = layers[li];

            // RMSNorm 1
            rms_norm(x.data(), L.inputNorm.data(), H, RMS_EPS, xNorm.data());

            // QKV с bias
            matvec_parallel(xNorm.data(), H, L.q_w.data(), L.q_b.data(), q.data(), H);
            matvec_parallel(xNorm.data(), H, L.k_w.data(), L.k_b.data(), k.data(), KVD);
            matvec_parallel(xNorm.data(), H, L.v_w.data(), L.v_b.data(), v.data(), KVD);

            // RoPE
            for (int h = 0; h < NH; h++)
                apply_rope(q.data(), h * HD, HD, position, ROPE_THETA);
            for (int h = 0; h < NKV; h++)
                apply_rope(k.data(), h * HD, HD, position, ROPE_THETA);

            // KV cache
            kv_k[li].push_back(k);
            kv_v[li].push_back(v);
            int seqLen = (int)kv_k[li].size();

            // Attention (GQA)
            float scale = 1.0f / std::sqrt((float)HD);
            for (int h = 0; h < NH; h++) {
                int kvHead = h / GROUPS;
                int qOff = h * HD;
                int kvOff = kvHead * HD;

                for (int t = 0; t < seqLen; t++) {
                    const float* kt = kv_k[li][t].data() + kvOff;
                    float s = 0.0f;
                    for (int d = 0; d < HD; d++) s += q[qOff + d] * kt[d];
                    scores[t] = s * scale;
                }
                softmax_inplace(scores.data(), seqLen);

                for (int d = 0; d < HD; d++) {
                    float s = 0.0f;
                    for (int t = 0; t < seqLen; t++)
                        s += scores[t] * kv_v[li][t][kvOff + d];
                    attnOut[qOff + d] = s;
                }
            }

            // o_proj + residual
            matvec_parallel(attnOut.data(), H, L.o_w.data(), nullptr, oProj.data(), H);
            for (int i = 0; i < H; i++) x[i] += oProj[i];

            // RMSNorm 2
            rms_norm(x.data(), L.postNorm.data(), H, RMS_EPS, xNorm.data());

            // SwiGLU
            matvec_parallel(xNorm.data(), H, L.gate_w.data(), nullptr, gate.data(), INTER);
            matvec_parallel(xNorm.data(), H, L.up_w.data(), nullptr, up.data(), INTER);
            silu_inplace(gate.data(), INTER);
            for (int i = 0; i < INTER; i++) gate[i] *= up[i];
            matvec_parallel(gate.data(), INTER, L.down_w.data(), nullptr, mlp.data(), H);

            for (int i = 0; i < H; i++) x[i] += mlp[i];
        }

        // Final norm
        rms_norm(x.data(), finalNorm.data(), H, RMS_EPS, xNorm.data());

        // LM head (tied)
        matvec_parallel(xNorm.data(), H, embed.data(), nullptr, logits.data(), VOCAB);

        return logits.data();
    }
};