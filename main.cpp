#include "st.h"
#include "st_writer.h"
#include "config.h"
#include "tokenizer.h"
#include "model.h"
#include "agent.h"
#include "ui.h"

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
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
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

    FILE* open_write_bin(const std::string& path) {
#ifdef _WIN32
        FILE* f = nullptr;
        if (fopen_s(&f, path.c_str(), "wb") != 0) return nullptr;
        return f;
#else
        return std::fopen(path.c_str(), "wb");
#endif
    }

    std::vector<std::pair<std::string, int>>
        choose_specials(const ModelConfig& cfg) {

        if (cfg.vocab_size > 100000) {
            if (cfg.is_qwen3()) {
                return {
                    {"<|endoftext|>", 151643},
                    {"<|im_start|>",  151644},
                    {"<|im_end|>",    151645},
                    {"<think>",       151667},
                    {"</think>",      151668},
                };
            }
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
    ModelConfig mc = load_model_config(config_path);

    t0 = std::chrono::high_resolution_clock::now();
    Qwen2Tokenizer tok(argv[2], argv[3], choose_specials(mc));
    t1 = std::chrono::high_resolution_clock::now();
    { char b[64]; std::snprintf(b, sizeof(b), " (vocab=%d)", tok.vocab_size());
    log_elapsed("tokenizer", t0, t1, b); }

    t0 = std::chrono::high_resolution_clock::now();
    QwenModel model(st, mc);
    t1 = std::chrono::high_resolution_clock::now();
    log_elapsed("model", t0, t1);

    return run_agent(model, tok, mc);
}

int main(int argc, char** argv) {
    try { return airun_main(argc, argv); }
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