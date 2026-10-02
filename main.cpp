// cpp/main.cpp — single entry point with two modes:
//   airun <model.safetensors> <vocab.json> <merges.txt>       — chat
//   airun --quantize <input.safetensors> <output.safetensors> — quantize
#include "st.h"
#include "st_writer.h"
#include "tokenizer.h"
#include "model.h"
#include "agent.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>

namespace {

    void log_elapsed(const char* stage,
        std::chrono::high_resolution_clock::time_point t0,
        std::chrono::high_resolution_clock::time_point t1,
        const char* suffix = "") {
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "[load] %s: %.1f ms%s\n", stage, ms, suffix);
    }

    // ─── INT8 quantization helper ─────────────────────────────
    struct QTensor {
        std::vector<int8_t> q;
        std::vector<float>  scales;
        int rows = 0, cols = 0;
    };

    QTensor quantize_matrix(const std::vector<float>& fp32, int rows, int cols) {
        QTensor t;
        t.rows = rows;
        t.cols = cols;
        t.q.resize(static_cast<size_t>(rows) * cols);
        t.scales.resize(rows);

        for (int j = 0; j < rows; ++j) {
            const float* row = fp32.data() + static_cast<size_t>(j) * cols;
            float mx = 0.0f;
            for (int i = 0; i < cols; ++i) {
                const float a = std::fabs(row[i]);
                if (a > mx) mx = a;
            }
            float sc = mx / 127.0f;
            if (sc < 1e-9f) sc = 1e-9f;
            t.scales[j] = sc;

            const float inv = 1.0f / sc;
            int8_t* dst = t.q.data() + static_cast<size_t>(j) * cols;
            for (int i = 0; i < cols; ++i) {
                int32_t v = static_cast<int32_t>(std::lround(row[i] * inv));
                if (v > 127) v = 127;
                if (v < -127) v = -127;
                dst[i] = static_cast<int8_t>(v);
            }
        }
        return t;
    }

    // ─── Mode: --quantize ─────────────────────────────────────
    int run_quantize(const char* in_path, const char* out_path) {
        const auto t0 = std::chrono::high_resolution_clock::now();

        std::fprintf(stderr, "[quantize] reading %s\n", in_path);
        SafeTensors st(in_path);

        // Hardcoded dims for Qwen2.5-0.5B-Instruct.
        constexpr int H = 896;
        constexpr int KVD = 128;
        constexpr int INTER = 4864;
        constexpr int NL = 24;
        constexpr int VOCAB = 151936;

        SafeTensorsWriter w;

        // Global tensors (FP32).
        {
            auto t = SafeTensors::to_f32(st["model.embed_tokens.weight"]);
            w.add_f32("model.embed_tokens.weight", { VOCAB, H }, t.data(), t.size());
        }
        {
            auto t = SafeTensors::to_f32(st["model.norm.weight"]);
            w.add_f32("model.norm.weight", { H }, t.data(), t.size());
        }

        // Per-layer.
        for (int i = 0; i < NL; ++i) {
            const std::string p = "model.layers." + std::to_string(i) + ".";

            // Norms (FP32, tiny).
            auto n1 = SafeTensors::to_f32(st[p + "input_layernorm.weight"]);
            w.add_f32(p + "input_layernorm.weight", { H }, n1.data(), n1.size());

            auto n2 = SafeTensors::to_f32(st[p + "post_attention_layernorm.weight"]);
            w.add_f32(p + "post_attention_layernorm.weight", { H }, n2.data(), n2.size());

            // Linear weights → INT8 + scale.
            struct Spec { std::string name; int rows; int cols; };
            const Spec specs[] = {
                { p + "self_attn.q_proj.weight", H,     H     },
                { p + "self_attn.k_proj.weight", KVD,   H     },
                { p + "self_attn.v_proj.weight", KVD,   H     },
                { p + "self_attn.o_proj.weight", H,     H     },
                { p + "mlp.gate_proj.weight",    INTER, H     },
                { p + "mlp.up_proj.weight",      INTER, H     },
                { p + "mlp.down_proj.weight",    H,     INTER },
            };

            for (const auto& s : specs) {
                auto fp = SafeTensors::to_f32(st[s.name]);
                auto q = quantize_matrix(fp, s.rows, s.cols);
                w.add_i8(s.name, { s.rows, s.cols }, q.q.data(), q.q.size());
                w.add_f32(s.name + ".scale", { s.rows }, q.scales.data(), q.scales.size());
            }

            // Biases (FP32, tiny).
            auto qb = SafeTensors::to_f32(st[p + "self_attn.q_proj.bias"]);
            w.add_f32(p + "self_attn.q_proj.bias", { H }, qb.data(), qb.size());
            auto kb = SafeTensors::to_f32(st[p + "self_attn.k_proj.bias"]);
            w.add_f32(p + "self_attn.k_proj.bias", { KVD }, kb.data(), kb.size());
            auto vb = SafeTensors::to_f32(st[p + "self_attn.v_proj.bias"]);
            w.add_f32(p + "self_attn.v_proj.bias", { KVD }, vb.data(), vb.size());

            if (i % 4 == 3 || i == NL - 1) {
                std::fprintf(stderr, "[quantize] layer %d/%d\n", i + 1, NL);
            }
        }

        std::fprintf(stderr, "[quantize] writing %s...\n", out_path);
        if (!w.save(out_path)) {
            std::fprintf(stderr, "ERROR: cannot write output\n");
            return 1;
        }

        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "[quantize] done in %.1f ms\n", ms);
        return 0;
    }

} // namespace

int main(int argc, char** argv) {
    // ─── Mode: --quantize ─────────────────────────────────────
    if (argc >= 2 && std::strcmp(argv[1], "--quantize") == 0) {
        if (argc < 4) {
            std::fprintf(stderr,
                "usage: %s --quantize input.safetensors output.safetensors\n",
                argv[0]);
            return 1;
        }
        return run_quantize(argv[2], argv[3]);
    }

    // ─── Mode: chat (default) ─────────────────────────────────
    ui::init();

    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s model.safetensors vocab.json merges.txt\n"
            "       %s --quantize input.safetensors output.safetensors\n",
            argv[0], argv[0]);
        return 1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    SafeTensors st(argv[1]);
    auto t1 = std::chrono::high_resolution_clock::now();
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), " (%zu tensors)", st.size());
        log_elapsed("safetensors", t0, t1, buf);
    }

    t0 = std::chrono::high_resolution_clock::now();
    Qwen2Tokenizer tok(argv[2], argv[3]);
    t1 = std::chrono::high_resolution_clock::now();
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), " (vocab=%d)", tok.vocab_size());
        log_elapsed("tokenizer", t0, t1, buf);
    }

    t0 = std::chrono::high_resolution_clock::now();
    Qwen2Model model(st);
    t1 = std::chrono::high_resolution_clock::now();
    log_elapsed("model", t0, t1);

    run_agent(model, tok);

    return 0;
}