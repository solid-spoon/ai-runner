#pragma once
#include <cstdio>
#include <cctype>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

struct ModelConfig {
    std::string model_type;
    int hidden_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int num_hidden_layers = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1e6f;
    bool tie_word_embeddings = true;
    int eos_token_id = -1;
    bool use_qk_norm = false;

    bool is_qwen2() const { return model_type == "qwen2"; }
    bool is_qwen3() const { return model_type == "qwen3"; }
    bool is_minimind() const { return model_type == "minimind"; }

    int effective_head_dim() const {
        return head_dim > 0 ? head_dim : (hidden_size / num_attention_heads);
    }
    int effective_kv_heads() const {
        return num_key_value_heads > 0 ? num_key_value_heads : num_attention_heads;
    }
};

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
            skip_ws();
            expect(':');
            skip_ws();
            parse_value(key, cb);
            skip_ws();
            if (i_ < s_.size() && s_[i_] == ',') ++i_;
        }
    }

private:
    void skip_ws() {
        while (i_ < s_.size() && std::isspace((unsigned char)s_[i_])) ++i_;
    }

    void expect(char c) {
        if (i_ >= s_.size() || s_[i_] != c)
            throw std::runtime_error(std::string("expected '") + c + "' at " + std::to_string(i_));
        ++i_;
    }

    std::string read_string() {
        expect('"');
        std::string r;
        while (i_ < s_.size() && s_[i_] != '"') {
            if (s_[i_] == '\\' && i_ + 1 < s_.size()) {
                ++i_;
                switch (s_[i_]) {
                case 'n': r += '\n'; break;
                case 't': r += '\t'; break;
                case 'r': r += '\r'; break;
                case 'b': r += '\b'; break;
                case 'f': r += '\f'; break;
                case '"': r += '"'; break;
                case '\\': r += '\\'; break;
                case '/': r += '/'; break;
                default: r += s_[i_]; break;
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
            return;
        }
        if (s_.compare(i_, 4, "true") == 0) {
            i_ += 4; cb(key, "true", "bool"); return;
        }
        if (s_.compare(i_, 5, "false") == 0) {
            i_ += 5; cb(key, "false", "bool"); return;
        }
        if (s_.compare(i_, 4, "null") == 0) {
            i_ += 4; return;
        }

        const size_t start = i_;
        if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) ++i_;
        bool is_float = false;
        while (i_ < s_.size() && (std::isdigit((unsigned char)s_[i_]) ||
            s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E' ||
            s_[i_] == '+' || s_[i_] == '-')) {
            if (s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E') is_float = true;
            ++i_;
        }
        if (i_ > start) {
            cb(key, s_.substr(start, i_ - start), is_float ? "float" : "int");
        }
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

inline ModelConfig load_model_config(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open config: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();

    ModelConfig cfg;
    JsonParser p(s);
    p.parse_top_object([&](const std::string& k, const std::string& v, const std::string& t) {
        if (k == "model_type") cfg.model_type = v;
        else if (k == "hidden_size") cfg.hidden_size = std::stoi(v);
        else if (k == "num_attention_heads") cfg.num_attention_heads = std::stoi(v);
        else if (k == "num_key_value_heads") cfg.num_key_value_heads = std::stoi(v);
        else if (k == "head_dim") cfg.head_dim = std::stoi(v);
        else if (k == "num_hidden_layers") cfg.num_hidden_layers = std::stoi(v);
        else if (k == "intermediate_size") cfg.intermediate_size = std::stoi(v);
        else if (k == "vocab_size") cfg.vocab_size = std::stoi(v);
        else if (k == "rms_norm_eps") cfg.rms_norm_eps = std::stof(v);
        else if (k == "rope_theta") cfg.rope_theta = std::stof(v);
        else if (k == "tie_word_embeddings") cfg.tie_word_embeddings = (v == "true");
        else if (k == "eos_token_id") cfg.eos_token_id = std::stoi(v);
        });

    if (cfg.hidden_size <= 0 || cfg.num_hidden_layers <= 0 ||
        cfg.vocab_size <= 0 || cfg.num_attention_heads <= 0)
        throw std::runtime_error("config.json missing or invalid required fields");

    return cfg;
}