#include "st.h"
#include "tokenizer.h"
#include "model.h"
#include "agent.h"
#include <cstdio>
#include <chrono>
#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s model.safetensors vocab.json merges.txt\n", argv[0]);
        return 1;
    }

    // ─── 1. Веса ───
    auto t0 = std::chrono::high_resolution_clock::now();
    SafeTensors st(argv[1]);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "[load] safetensors: %.1f ms (%zu tensors)\n",
        std::chrono::duration<double, std::milli>(t1 - t0).count(),
        st.size());

    // ─── 2. Токенизатор ───
    t0 = std::chrono::high_resolution_clock::now();
    Qwen2Tokenizer tok(argv[2], argv[3]);
    t1 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "[load] tokenizer: %.1f ms (vocab=%d)\n",
        std::chrono::duration<double, std::milli>(t1 - t0).count(),
        tok.vocab_size());

    // ─── 3. Модель (BF16 → FP32 в RAM) ───
    t0 = std::chrono::high_resolution_clock::now();
    Qwen2Model model(st);
    t1 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "[load] model FP32: %.1f ms\n",
        std::chrono::duration<double, std::milli>(t1 - t0).count());

    // ─── 4. Интерактивный агент ───
    run_agent(model, tok);

    return 0;
}