// cpp/agent.h — Qwen2 chat agent: prompt template + generation loop
#pragma once
#include "model.h"
#include "tokenizer.h"
#include "sampler.h"
#include <cstdio>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <sstream>

// Спец-токены Qwen2
static constexpr int TOK_IM_START = 151644;
static constexpr int TOK_IM_END = 151645;
static constexpr int TOK_ENDOFTEXT = 151643;

struct ChatTurn {
    std::string role;   // "user" / "assistant"
    std::string text;
};

class QwenAgent {
    Qwen2Model& model_;
    Qwen2Tokenizer& tok_;
    Sampler sampler_;
    SamplerConfig cfg_;

    std::vector<ChatTurn> history_;

public:
    QwenAgent(Qwen2Model& m, Qwen2Tokenizer& t) : model_(m), tok_(t) {}

    void set_config(const SamplerConfig& c) { cfg_ = c; }
    void clear_history() { history_.clear(); }

    // Собираем промпт в формате Qwen2 chat
    std::string build_prompt(const std::string& system,
        const std::string& user_input) {
        std::ostringstream s;
        s << "<|im_start|>system\n" << system << "<|im_end|>\n";
        for (auto& t : history_) {
            s << "<|im_start|>" << t.role << "\n" << t.text << "<|im_end|>\n";
        }
        s << "<|im_start|>user\n" << user_input << "<|im_end|>\n";
        s << "<|im_start|>assistant\n";
        return s.str();
    }

    // Одна генерация. Возвращает сгенерированный текст (без спец-токенов).
    std::string generate(const std::string& system_prompt,
        const std::string& user_input,
        int max_tokens = 256) {
        std::string prompt = build_prompt(system_prompt, user_input);
        auto ids = tok_.encode(prompt);

        model_.reset_kv();

        // ??? PREFILL: прогоняем весь промпт ???
        const float* logits = nullptr;
        auto t0 = std::chrono::high_resolution_clock::now();

        std::fprintf(stderr, "[prefill] %zu tokens...", ids.size());
        std::fflush(stderr);

        for (size_t i = 0; i < ids.size(); i++)
            logits = model_.forward(ids[i], (int)i);

        auto t1 = std::chrono::high_resolution_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, " done in %.0f ms\n", prefill_ms);
        std::fflush(stderr);

        // ??? GENERATION ???
        std::vector<int> generated;
        std::vector<int> recent = ids;   // для repetition penalty
        int pos = (int)ids.size();

        auto g0 = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < max_tokens; step++) {
            int next = sampler_.sample(logits, model_.VOCAB, cfg_, recent);

            // Стоп по <|im_end|>
            if (next == TOK_IM_END || next == TOK_ENDOFTEXT) break;

            generated.push_back(next);
            recent.push_back(next);

            // Стрим: декодируем новый токен отдельно и печатаем сразу
            std::string chunk = tok_.decode({ next });
            std::fputs(chunk.c_str(), stdout);
            std::fflush(stdout);

            logits = model_.forward(next, pos++);
        }

        auto g1 = std::chrono::high_resolution_clock::now();
        double gen_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
        double tps = generated.empty() ? 0 : generated.size() * 1000.0 / gen_ms;
        std::fprintf(stderr, "\n[gen] %zu tokens in %.0f ms (%.1f tok/s)\n",
            generated.size(), gen_ms, tps);

        // Записываем в историю
        history_.push_back({ "user", user_input });
        history_.push_back({ "assistant", tok_.decode(generated) });

        // Обрезаем историю до ~6 реплик (3 пары)
        while (history_.size() > 6) history_.erase(history_.begin());

        return tok_.decode(generated);
    }
};

// ??? Интерактивный REPL ???
inline void run_agent(Qwen2Model& model, Qwen2Tokenizer& tok) {
    QwenAgent agent(model, tok);

    const std::string SYSTEM =
        "You are a helpful, concise AI assistant. "
        "Answer in the same language the user writes in.";

    std::printf("\n");
    std::printf("========================================\n");
    std::printf("  Qwen2.5-0.5B Agent (C++ native)\n");
    std::printf("  Commands: /reset  /exit  /temp N\n");
    std::printf("========================================\n\n");

    SamplerConfig cfg;
    cfg.temperature = 0.7f;
    cfg.top_k = 20;
    cfg.top_p = 0.8f;
    cfg.repetition_penalty = 1.05f;
    agent.set_config(cfg);

    std::string line;
    while (true) {
        std::printf("you > ");
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) break;

        // trim
        size_t a = line.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        size_t b = line.find_last_not_of(" \t\r\n");
        line = line.substr(a, b - a + 1);

        if (line == "/exit" || line == "/quit") break;
        if (line == "/reset") {
            agent.clear_history();
            std::printf("[history cleared]\n\n");
            continue;
        }
        if (line.rfind("/temp ", 0) == 0) {
            try {
                float t = std::stof(line.substr(6));
                cfg.temperature = t;
                agent.set_config(cfg);
                std::printf("[temperature = %.2f]\n\n", t);
            }
            catch (...) {
                std::printf("[usage: /temp 0.7]\n\n");
            }
            continue;
        }

        std::printf("ai  > ");
        std::fflush(stdout);

        agent.generate(SYSTEM, line, 256);

        std::printf("\n\n");
    }

    std::printf("bye.\n");
}