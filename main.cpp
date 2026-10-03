// cpp/main.cpp — single entry point.
#include "st.h"
#include "st_writer.h"
#include "config.h"
#include "tokenizer.h"
#include "model.h"
#include "agent.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>

namespace {

    void log_elapsed(const char* stage,
        std::chrono::high_resolution_clock::time_point t0,
        std::chrono::high_resolution_clock::time_point t1,
        const char* suffix = "") {
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "[load] %s: %.1f ms%s\n", stage, ms, suffix);
    }

    std::string dir_of(const std::string& path) {
        size_t slash = path.find_last_of("/\\");
        return (slash == std::string::npos) ? std::string(".") : path.substr(0, slash);
    }

    bool file_exists(const std::string& p) {
        std::ifstream f(p, std::ios::binary);
        return static_cast<bool>(f);
    }

    size_t file_size(const std::string& p) {
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f) return 0;
        return static_cast<size_t>(f.tellg());
    }

    std::vector<std::pair<std::string, int>>
        choose_specials(const ModelConfig& cfg) {
        if (cfg.is_qwen3()) {
            return {
                {"<|endoftext|>", 151643},
                {"<|im_start|>",  151644},
                {"<|im_end|>",    151645},
                {"<think>",       151667},
                {"</think>",      151668},
            };
        }
        if (cfg.vocab_size > 100000) {
            return {
                {"<|endoftext|>", 151643},
                {"<|im_start|>",  151644},
                {"<|im_end|>",    151645},
            };
        }
        return {
            {"<|endoftext|>",         0},
            {"<|im_start|>",          1},
            {"<|im_end|>",            2},
            {"<|object_ref_start|>",  3},
            {"<|object_ref_end|>",    4},
            {"<|box_start|>",         5},
            {"<|box_end|>",           6},
            {"<|quad_start|>",        7},
            {"<|quad_end|>",          8},
            {"<|vision_start|>",      9},
            {"<|vision_end|>",        10},
            {"<|vision_pad|>",        11},
            {"<|image_pad|>",         12},
            {"<|video_pad|>",         13},
            {"<|audio_start|>",       14},
            {"<|audio_end|>",         15},
            {"<|audio_pad|>",         16},
            {"<tts_pad>",             17},
            {"<tts_text_bos>",        18},
            {"<tts_text_eod>",        19},
            {"<tts_text_bos_single>", 20},
            {"<tool_call>",           21},
            {"</tool_call>",          22},
            {"<tool_response>",       23},
            {"</tool_response>",      24},
            {"<think>",               25},
            {"</think>",              26},
            {"<|buffer1|>",           27},
            {"<|buffer2|>",           28},
            {"<|buffer3|>",           29},
            {"<|buffer4|>",           30},
            {"<|buffer5|>",           31},
            {"<|buffer6|>",           32},
            {"<|buffer7|>",           33},
            {"<|buffer8|>",           34},
            {"<|buffer9|>",           35},
        };
    }

} // namespace

static int airun_main(int argc, char** argv) {
    std::fprintf(stderr, "[main] argc=%d\n", argc);
    for (int i = 0; i < argc; ++i)
        std::fprintf(stderr, "[main]   argv[%d] = \"%s\"\n", i, argv[i]);

    // ─── --list ──────────────────────────────────────────────
    if (argc >= 2 && std::strcmp(argv[1], "--list") == 0) {
        if (argc < 3) {
            std::fprintf(stderr, "usage: %s --list model.safetensors\n", argv[0]);
            return 1;
        }
        if (!file_exists(argv[2]))
            throw std::runtime_error(std::string("file does not exist: ") + argv[2]);
        SafeTensors st(argv[2]);
        std::printf("Tensors: %zu\n\n", st.size());
        std::vector<std::string> keys;
        for (const auto& kv : st.all()) keys.push_back(kv.first);
        std::sort(keys.begin(), keys.end());
        size_t total = 0;
        for (const auto& k : keys) {
            const TensorInfo& t = st[k];
            total += t.nbytes();
            std::printf("  %-60s %-5s [", k.c_str(), t.dtype.c_str());
            for (size_t i = 0; i < t.shape.size(); ++i)
                std::printf("%ld%s", (long)t.shape[i], i + 1 < t.shape.size() ? "," : "");
            std::printf("]\n");
        }
        std::printf("\nTotal: %.2f MB\n", total / 1e6);
        return 0;
    }

    // ─── --extract-tokenizer ─────────────────────────────────
    if (argc >= 2 && std::strcmp(argv[1], "--extract-tokenizer") == 0) {
        if (argc < 4) {
            std::fprintf(stderr,
                "usage: %s --extract-tokenizer tokenizer.json out_dir\n",
                argv[0]);
            return 1;
        }
        std::ifstream f(argv[2], std::ios::binary);
        if (!f)
            throw std::runtime_error(std::string("cannot open: ") + argv[2]);
        std::stringstream ss; ss << f.rdbuf();
        const std::string s = ss.str();
        const std::string out_dir = argv[3];

        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);
        if (ec)
            throw std::runtime_error("cannot create output dir: " + out_dir + " (" + ec.message() + ")");

        auto find_key = [&](const char* key) -> size_t {
            std::string needle = std::string("\"") + key + "\":";
            return s.find(needle);
        };

        auto extract_container = [&](size_t start) -> std::pair<std::string, size_t> {
            const char open = s[start];
            const char close = (open == '{') ? '}' : ']';
            int depth = 1;
            bool in_str = false;
            size_t pos = start + 1;
            while (pos < s.size()) {
                char c = s[pos];
                if (in_str) {
                    if (c == '\\') { pos += 2; continue; }
                    if (c == '"') in_str = false;
                }
                else {
                    if (c == '"') in_str = true;
                    else if (c == open) depth++;
                    else if (c == close) {
                        depth--;
                        if (depth == 0) { pos++; break; }
                    }
                }
                pos++;
            }
            return { s.substr(start, pos - start), pos };
        };

        size_t vk = find_key("vocab");
        if (vk == std::string::npos)
            throw std::runtime_error("tokenizer.json: no \"vocab\" field");
        size_t vb = s.find('{', vk);
        if (vb == std::string::npos)
            throw std::runtime_error("tokenizer.json: no '{' after vocab");
        auto vpair = extract_container(vb);
        const std::string& vocab_obj = vpair.first;
        const std::string vocab_path = out_dir + "/vocab.json";
        {
            FILE* o = nullptr;
            if (fopen_s(&o, vocab_path.c_str(), "wb") != 0 || !o)
                throw std::runtime_error("cannot write: " + vocab_path);
            std::fwrite(vocab_obj.data(), 1, vocab_obj.size(), o);
            std::fclose(o);
        }
        std::fprintf(stderr, "[extract] wrote %s (%zu bytes)\n",
            vocab_path.c_str(), vocab_obj.size());

        size_t mk = find_key("merges");
        if (mk == std::string::npos)
            throw std::runtime_error("tokenizer.json: no \"merges\" field");
        size_t mb = s.find('[', mk);
        if (mb == std::string::npos)
            throw std::runtime_error("tokenizer.json: no '[' after merges");
        auto mpair = extract_container(mb);
        const std::string& mjson = mpair.first;

        auto read_str = [](const std::string& src, size_t& p) -> std::string {
            std::string r;
            if (p >= src.size() || src[p] != '"') return r;
            p++;
            while (p < src.size() && src[p] != '"') {
                if (src[p] == '\\' && p + 1 < src.size()) {
                    char e = src[p + 1];
                    if (e == 'n')  r += '\n';
                    else if (e == 't')  r += '\t';
                    else if (e == 'r')  r += '\r';
                    else if (e == 'b')  r += '\b';
                    else if (e == 'f')  r += '\f';
                    else if (e == '"')  r += '"';
                    else if (e == '\\') r += '\\';
                    else if (e == '/')  r += '/';
                    else                r += e;
                    p += 2;
                }
                else {
                    r += src[p++];
                }
            }
            if (p < src.size()) p++;
            return r;
        };

        std::string merges_txt = "#version: 0.2\n";
        size_t i = 1;
        size_t n_merges = 0;
        while (i < mjson.size()) {
            while (i < mjson.size() &&
                (mjson[i] == ' ' || mjson[i] == '\n' || mjson[i] == '\r' ||
                    mjson[i] == '\t' || mjson[i] == ','))
                i++;
            if (i >= mjson.size() || mjson[i] == ']') break;

            if (mjson[i] == '[') {
                i++;
                while (i < mjson.size() &&
                    (mjson[i] == ' ' || mjson[i] == '\n' || mjson[i] == '\r' || mjson[i] == '\t'))
                    i++;
                std::string a = read_str(mjson, i);
                while (i < mjson.size() &&
                    (mjson[i] == ' ' || mjson[i] == '\n' || mjson[i] == '\r' ||
                        mjson[i] == '\t' || mjson[i] == ','))
                    i++;
                std::string b = read_str(mjson, i);
                while (i < mjson.size() && mjson[i] != ']') i++;
                if (i < mjson.size()) i++;
                merges_txt += a;
                merges_txt += ' ';
                merges_txt += b;
                merges_txt += '\n';
                n_merges++;
            }
            else if (mjson[i] == '"') {
                std::string a = read_str(mjson, i);
                merges_txt += a;
                merges_txt += '\n';
                n_merges++;
            }
            else {
                i++;
            }
        }

        const std::string merges_path = out_dir + "/merges.txt";
        {
            FILE* o = nullptr;
            if (fopen_s(&o, merges_path.c_str(), "wb") != 0 || !o)
                throw std::runtime_error("cannot write: " + merges_path);
            std::fwrite(merges_txt.data(), 1, merges_txt.size(), o);
            std::fclose(o);
        }
        std::fprintf(stderr, "[extract] wrote %s (%zu merges)\n",
            merges_path.c_str(), n_merges);
        return 0;
    }

    // ─── --quantize (config-driven) ──────────────────────────
    if (argc >= 2 && std::strcmp(argv[1], "--quantize") == 0) {
        if (argc < 4) {
            std::fprintf(stderr,
                "usage: %s --quantize in.safetensors out.safetensors\n",
                argv[0]);
            return 1;
        }
        const std::string in_path = argv[2];
        const std::string out_path = argv[3];
        const std::string cfg_path = dir_of(in_path) + "/config.json";

        if (!file_exists(in_path))
            throw std::runtime_error("input file does not exist: " + in_path);
        const size_t in_size = file_size(in_path);
        std::fprintf(stderr, "[quantize] input: %s (%.2f MB)\n",
            in_path.c_str(), in_size / 1e6);

        if (!file_exists(cfg_path))
            throw std::runtime_error("config.json not found next to input: " + cfg_path);
        std::fprintf(stderr, "[quantize] config: %s\n", cfg_path.c_str());

        std::error_code ec;
        const std::string out_dir = dir_of(out_path);
        std::filesystem::create_directories(out_dir, ec);
        if (ec)
            throw std::runtime_error("cannot create output dir: " + out_dir + " (" + ec.message() + ")");
        std::fprintf(stderr, "[quantize] output dir ready: %s\n", out_dir.c_str());

        SafeTensors st(in_path);
        ModelConfig mc = load_model_config(cfg_path);

        const int H = mc.hidden_size;
        const int NH = mc.num_attention_heads;
        const int NKV = mc.effective_kv_heads();
        const int HD = mc.effective_head_dim();
        const int Q_DIM = NH * HD;
        const int KVD = NKV * HD;
        const int NL = mc.num_hidden_layers;
        const int INTER = mc.intermediate_size;
        const int VOCAB = mc.vocab_size;

        std::fprintf(stderr,
            "[quantize] dims: H=%d NH=%d NKV=%d HD=%d Q_DIM=%d KVD=%d NL=%d INTER=%d VOCAB=%d\n",
            H, NH, NKV, HD, Q_DIM, KVD, NL, INTER, VOCAB);

        SafeTensorsWriter w;

        auto shape_str = [](const TensorInfo& t) {
            std::string s = "[";
            for (size_t i = 0; i < t.shape.size(); ++i) {
                if (i) s += ",";
                s += std::to_string(t.shape[i]);
            }
            s += "]";
            return s;
        };

        auto qmat = [](const std::vector<float>& fp32, int rows, int cols) {
            std::vector<int8_t> q((size_t)rows * cols);
            std::vector<float>  sc(rows);
            for (int j = 0; j < rows; ++j) {
                const float* row = fp32.data() + (size_t)j * cols;
                float mx = 0.0f;
                for (int i = 0; i < cols; ++i) {
                    const float a = std::fabs(row[i]);
                    if (a > mx) mx = a;
                }
                float s = mx / 127.0f;
                if (s < 1e-9f) s = 1e-9f;
                sc[j] = s;
                int8_t* dst = q.data() + (size_t)j * cols;
                for (int i = 0; i < cols; ++i) {
                    int32_t v = (int32_t)std::lround(row[i] / s);
                    if (v > 127) v = 127;
                    if (v < -127) v = -127;
                    dst[i] = (int8_t)v;
                }
            }
            return std::make_pair(q, sc);
        };

        auto quantize_linear = [&](const std::string& name, int rows, int cols) {
            if (!st.has(name))
                throw std::runtime_error("quantize: missing tensor " + name);
            const TensorInfo& info = st[name];
            if ((int)info.shape.size() != 2 ||
                (int)info.shape[0] != rows || (int)info.shape[1] != cols) {
                throw std::runtime_error(
                    "quantize: tensor '" + name + "' has shape " + shape_str(info) +
                    ", expected [" + std::to_string(rows) + "," + std::to_string(cols) + "]");
            }
            auto t = SafeTensors::to_f32(info);
            auto [q, sc] = qmat(t, rows, cols);
            w.add_i8(name, { rows, cols }, q.data(), q.size());
            w.add_f32(name + ".scale", { rows }, sc.data(), sc.size());
        };

        auto copy_f32 = [&](const std::string& name) {
            if (!st.has(name))
                throw std::runtime_error("quantize: missing tensor " + name);
            const TensorInfo& info = st[name];
            auto t = SafeTensors::to_f32(info);
            w.add_f32(name, { (int64_t)t.size() }, t.data(), t.size());
        };

        std::fprintf(stderr, "[quantize] embedding...\n");
        quantize_linear("model.embed_tokens.weight", VOCAB, H);

        std::fprintf(stderr, "[quantize] final norm...\n");
        copy_f32("model.norm.weight");

        for (int i = 0; i < NL; ++i) {
            const std::string p = "model.layers." + std::to_string(i) + ".";

            copy_f32(p + "input_layernorm.weight");
            copy_f32(p + "post_attention_layernorm.weight");

            if (st.has(p + "self_attn.q_norm.weight")) {
                copy_f32(p + "self_attn.q_norm.weight");
                copy_f32(p + "self_attn.k_norm.weight");
            }

            quantize_linear(p + "self_attn.q_proj.weight", Q_DIM, H);
            quantize_linear(p + "self_attn.k_proj.weight", KVD, H);
            quantize_linear(p + "self_attn.v_proj.weight", KVD, H);
            quantize_linear(p + "self_attn.o_proj.weight", H, Q_DIM);
            quantize_linear(p + "mlp.gate_proj.weight", INTER, H);
            quantize_linear(p + "mlp.up_proj.weight", INTER, H);
            quantize_linear(p + "mlp.down_proj.weight", H, INTER);

            if (st.has(p + "self_attn.q_proj.bias")) {
                copy_f32(p + "self_attn.q_proj.bias");
                copy_f32(p + "self_attn.k_proj.bias");
                copy_f32(p + "self_attn.v_proj.bias");
            }

            if ((i + 1) % 4 == 0 || i == NL - 1)
                std::fprintf(stderr, "[quantize] layer %d/%d\n", i + 1, NL);
        }

        std::fprintf(stderr, "[quantize] writing %s...\n", out_path.c_str());
        if (!w.save(out_path))
            throw std::runtime_error("cannot write output: " + out_path);

        const size_t out_size = file_size(out_path);
        std::fprintf(stderr, "[quantize] done. output: %.2f MB (ratio %.2fx)\n",
            out_size / 1e6, (double)in_size / (double)out_size);
        return 0;
    }

    // ─── Chat mode ───────────────────────────────────────────
    ui::init();

    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s model.safetensors vocab.json merges.txt\n"
            "       %s --quantize in.safetensors out.safetensors\n"
            "       %s --list model.safetensors\n"
            "       %s --extract-tokenizer tokenizer.json out_dir\n",
            argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }

    if (!file_exists(argv[1]))
        throw std::runtime_error(std::string("model file does not exist: ") + argv[1]);
    if (!file_exists(argv[2]))
        throw std::runtime_error(std::string("vocab file does not exist: ") + argv[2]);
    if (!file_exists(argv[3]))
        throw std::runtime_error(std::string("merges file does not exist: ") + argv[3]);

    const size_t model_sz = file_size(argv[1]);
    std::fprintf(stderr, "[main] model file: %.2f MB\n", model_sz / 1e6);

    auto t0 = std::chrono::high_resolution_clock::now();
    SafeTensors st(argv[1]);
    auto t1 = std::chrono::high_resolution_clock::now();
    { char b[64]; std::snprintf(b, sizeof(b), " (%zu tensors)", st.size());
    log_elapsed("safetensors", t0, t1, b); }

    const std::string config_path = dir_of(argv[1]) + "/config.json";
    ModelConfig mc;
    try {
        mc = load_model_config(config_path);
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "[config] %s — falling back to Qwen2.5-0.5B defaults\n", e.what());
        mc.model_type = "qwen2";
        mc.hidden_size = 896; mc.num_attention_heads = 14; mc.num_key_value_heads = 2;
        mc.num_hidden_layers = 24; mc.intermediate_size = 4864; mc.vocab_size = 151936;
        mc.rms_norm_eps = 1e-6f; mc.rope_theta = 1e6f; mc.tie_word_embeddings = true;
    }

    t0 = std::chrono::high_resolution_clock::now();
    Qwen2Tokenizer tok(argv[2], argv[3], choose_specials(mc));
    t1 = std::chrono::high_resolution_clock::now();
    { char b[64]; std::snprintf(b, sizeof(b), " (vocab=%d)", tok.vocab_size());
    log_elapsed("tokenizer", t0, t1, b); }

    t0 = std::chrono::high_resolution_clock::now();
    QwenModel model(st, mc);
    t1 = std::chrono::high_resolution_clock::now();
    log_elapsed("model", t0, t1);

    run_agent(model, tok, mc);
    return 0;
}

int main(int argc, char** argv) {
    try {
        return airun_main(argc, argv);
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "\n[FATAL] %s\n", e.what());
        std::fflush(stderr);
        return 2;
    }
    catch (...) {
        std::fprintf(stderr, "\n[FATAL] unknown exception\n");
        std::fflush(stderr);
        return 2;
    }
}