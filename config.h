// cpp/config.h — minimal JSON parser for model config.json files.
//
// Supports Qwen2, Qwen3, MiniMind-style configs. Fields are extracted by
// name; unrecognized keys are ignored.
#pragma once

#include <cstdio>
#include <cstring>
#include <cctype>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

// ─── Parsed config ─────────────────────────────────────────
struct ModelConfig {
    std::string model_type;         // "qwen2" | "qwen3" | "minimind"

    int   hidden_size = 0;
    int   num_attention_heads = 0;
    int   num_key_value_heads = 0;   // 0 → defaults to num_attention_heads
    int   head_dim = 0;   // 0 → defaults to hidden_size / n_heads
    int   num_hidden_layers = 0;
    int   intermediate_size = 0;
    int   vocab_size = 0;
    int   max_position_embeddings = 32768;

    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;

    bool  tie_word_embeddings = true;
    bool  attention_bias = false;

    int   bos_token_id = -1;
    int   eos_token_id = -1;

    // Derived: does this config use QK-norm?
    // Filled by the loader based on presence of "qk_norm" hint or
    // detection of q_norm.weight in the safetensors file.
    bool  use_qk_norm = false;

    bool is_qwen2()    const { return model_type == "qwen2"; }
    bool is_qwen3()    const { return model_type == "qwen3"; }
    bool is_minimind() const { return model_type == "minimind"; }

    int  effective_head_dim() const {
        return head_dim > 0 ? head_dim : (hidden_size / num_attention_heads);
    }
    int  effective_kv_heads() const {
        return num_key_value_heads > 0 ? num_key_value_heads
            : num_attention_heads;
    }
};

// ─── Minimal JSON value walker ─────────────────────────────
class JsonParser {
    const std::string& s_;
    size_t i_ = 0;

public:
    explicit JsonParser(const std::string& s) : s_(s) {}

    template <typename Cb>
    void parse_top_object(Cb cb) {
        skip_ws();
        expect('{');
        while (true) {
            skip_ws();
            if (i_ >= s_.size()) break;
            if (s_[i_] == '}') { ++i_; break; }
            std::string key = read_string();
            skip_ws(); expect(':'); skip_ws();
            parse_value(key, cb);
            skip_ws();
            if (i_ < s_.size() && s_[i_] == ',') ++i_;
        }
    }

private:
    void skip_ws() {
        while (i_ < s_.size() &&
            (s_[i_] == ' ' || s_[i_] == '\n' ||
                s_[i_] == '\r' || s_[i_] == '\t')) ++i_;
    }
    void expect(char c) {
        if (i_ >= s_.size() || s_[i_] != c)
            throw std::runtime_error(std::string("expected '") + c +
                "' at " + std::to_string(i_));
        ++i_;
    }
    std::string read_string() {
        expect('"');
        std::string r;
        while (i_ < s_.size() && s_[i_] != '"') {
            if (s_[i_] == '\\' && i_ + 1 < s_.size()) {
                ++i_;
                switch (s_[i_]) {
                case 'n':  r += '\n'; break;
                case 't':  r += '\t'; break;
                case 'r':  r += '\r'; break;
                case '"':  r += '"';  break;
                case '\\': r += '\\'; break;
                case '/':  r += '/';  break;
                default:   r += s_[i_]; break;
                }
                ++i_;
            }
            else {
                r += s_[i_++];
            }
        }
        expect('"');
        return r;
    }

    template <typename Cb>
    void parse_value(const std::string& key, Cb& cb) {
        skip_ws();
        if (i_ >= s_.size()) return;
        const char c = s_[i_];

        if (c == '"') {
            cb(key, read_string(), "string");
            return;
        }
        if (c == '{' || c == '[') {
            skip_container();
            cb(key, "", "other");
            return;
        }
        if (std::strncmp(s_.c_str() + i_, "true", 4) == 0) {
            i_ += 4; cb(key, "true", "bool"); return;
        }
        if (std::strncmp(s_.c_str() + i_, "false", 5) == 0) {
            i_ += 5; cb(key, "false", "bool"); return;
        }
        if (std::strncmp(s_.c_str() + i_, "null", 4) == 0) {
            i_ += 4; cb(key, "", "null"); return;
        }
        const size_t start = i_;
        if (s_[i_] == '-' || s_[i_] == '+') ++i_;
        bool is_float = false;
        while (i_ < s_.size() &&
            (std::isdigit((unsigned char)s_[i_]) ||
                s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E' ||
                s_[i_] == '+' || s_[i_] == '-')) {
            if (s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E')
                is_float = true;
            ++i_;
        }
        cb(key, s_.substr(start, i_ - start), is_float ? "float" : "int");
    }

    void skip_container() {
        const char open = s_[i_];
        const char close = (open == '{') ? '}' : ']';
        int depth = 0;
        bool in_str = false;
        while (i_ < s_.size()) {
            const char ch = s_[i_];
            if (in_str) {
                if (ch == '\\') { i_ += 2; continue; }
                if (ch == '"') in_str = false;
            }
            else {
                if (ch == '"') in_str = true;
                else if (ch == open) ++depth;
                else if (ch == close) {
                    --depth;
                    if (depth == 0) { ++i_; return; }
                }
            }
            ++i_;
        }
    }
};

// ─── Load config.json ──────────────────────────────────────
inline ModelConfig load_model_config(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open config: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();

    ModelConfig cfg;
    JsonParser p(s);

    p.parse_top_object([&](const std::string& k, const std::string& v,
        const std::string& t) {
            auto to_i = [&]() { return std::stoi(v); };
            auto to_f = [&]() { return std::stof(v); };
            auto to_b = [&]() { return v == "true"; };

            if (k == "model_type")               cfg.model_type = v;
            else if (k == "hidden_size")              cfg.hidden_size = to_i();
            else if (k == "num_attention_heads")      cfg.num_attention_heads = to_i();
            else if (k == "num_key_value_heads")      cfg.num_key_value_heads = to_i();
            else if (k == "head_dim")                 cfg.head_dim = to_i();
            else if (k == "num_hidden_layers")        cfg.num_hidden_layers = to_i();
            else if (k == "intermediate_size")        cfg.intermediate_size = to_i();
            else if (k == "vocab_size")               cfg.vocab_size = to_i();
            else if (k == "max_position_embeddings")  cfg.max_position_embeddings = to_i();
            else if (k == "rms_norm_eps")             cfg.rms_norm_eps = to_f();
            else if (k == "rope_theta")               cfg.rope_theta = to_f();
            else if (k == "tie_word_embeddings")      cfg.tie_word_embeddings = to_b();
            else if (k == "attention_bias")           cfg.attention_bias = to_b();
            else if (k == "bos_token_id")             cfg.bos_token_id = to_i();
            else if (k == "eos_token_id")             cfg.eos_token_id = to_i();
        });

    if (cfg.hidden_size == 0 || cfg.num_hidden_layers == 0 ||
        cfg.vocab_size == 0) {
        throw std::runtime_error("config.json missing required fields");
    }

    std::fprintf(stderr,
        "[config] type=%s H=%d NH=%d NKV=%d HD=%d NL=%d INTER=%d VOCAB=%d "
        "eps=%.0e rope=%.0e tie=%d attn_bias=%d\n",
        cfg.model_type.c_str(),
        cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
        cfg.effective_head_dim(), cfg.num_hidden_layers,
        cfg.intermediate_size, cfg.vocab_size,
        cfg.rms_norm_eps, cfg.rope_theta,
        cfg.tie_word_embeddings ? 1 : 0, cfg.attention_bias ? 1 : 0);

    return cfg;
}