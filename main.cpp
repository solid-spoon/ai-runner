// cpp/main.cpp — entry point: load model + tokenizer, then run the REPL.
//
// IMPORTANT for MSVC users: compile with `/utf-8`. Without it, any non-ASCII
// string literals (including the Chinese text in the system prompt) will be
// re-encoded according to the active code page and will reach the model as
// garbage. GCC/Clang default to UTF-8 and need no flag.
//
#include "st.h"
#include "tokenizer.h"
#include "model.h"
#include "agent.h"

#include <cstdio>
#include <chrono>

namespace {

    // Prints a load-stage duration in milliseconds.
    void log_elapsed(const char* stage,
        std::chrono::high_resolution_clock::time_point t0,
        std::chrono::high_resolution_clock::time_point t1,
        const char* suffix = "") {
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "[load] %s: %.1f ms%s\n", stage, ms, suffix);
    }

} // namespace

int main(int argc, char** argv) {
    // Sets console code page, enables VT escape processing, and puts the
    // CRT streams into binary mode. See ui::init() for the details.
    ui::init();

    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s model.safetensors vocab.json merges.txt\n", argv[0]);
        return 1;
    }

    // 1. Weights.
    auto t0 = std::chrono::high_resolution_clock::now();
    SafeTensors st(argv[1]);
    auto t1 = std::chrono::high_resolution_clock::now();
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), " (%zu tensors)", st.size());
        log_elapsed("safetensors", t0, t1, buf);
    }

    // 2. Tokenizer.
    t0 = std::chrono::high_resolution_clock::now();
    Qwen2Tokenizer tok(argv[2], argv[3]);
    t1 = std::chrono::high_resolution_clock::now();
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), " (vocab=%d)", tok.vocab_size());
        log_elapsed("tokenizer", t0, t1, buf);
    }

    // 3. Model (BF16 weights dequantized to FP32 in RAM).
    t0 = std::chrono::high_resolution_clock::now();
    Qwen2Model model(st);
    t1 = std::chrono::high_resolution_clock::now();
    log_elapsed("model FP32", t0, t1);

    // 4. Interactive agent.
    run_agent(model, tok);

    return 0;
}