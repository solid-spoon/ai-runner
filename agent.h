// cpp/agent.h — Qwen2 chat agent driving the TUI.
#pragma once

#include "model.h"
#include "tokenizer.h"
#include "sampler.h"
#include "ui.h"

#include <cstdio>
#include <string>
#include <vector>
#include <chrono>
#include <sstream>

namespace tok_id {
    inline constexpr int kImStart = 151644;
    inline constexpr int kImEnd = 151645;
    inline constexpr int kEndOfText = 151643;
}

struct ChatTurn {
    std::string role;
    std::string text;
};

class QwenAgent {
public:
    QwenAgent(Qwen2Model& model, Qwen2Tokenizer& tok)
        : model_(model), tok_(tok) {}

    void set_config(const SamplerConfig& c) { cfg_ = c; }
    const SamplerConfig& config() const { return cfg_; }
    void clear_history() { history_.clear(); }

    // Builds the ChatML prompt for the given system + user input,
    // appending the full conversation history.
    std::string build_prompt(const std::string& system,
        const std::string& user_input) const {
        std::ostringstream s;
        s << "<|im_start|>system\n" << system << "<|im_end|>\n";
        for (const auto& t : history_)
            s << "<|im_start|>" << t.role << "\n" << t.text << "<|im_end|>\n";
        s << "<|im_start|>user\n" << user_input << "<|im_end|>\n";
        s << "<|im_start|>assistant\n";
        return s.str();
    }

    // Runs prefill on the prompt, then autoregressively samples up to
    // max_tokens new tokens. Streams the output to the UI.
    std::string generate(const std::string& system_prompt,
        const std::string& user_input,
        int max_tokens = 1024) {
        const std::string prompt = build_prompt(system_prompt, user_input);
        const auto ids = tok_.encode(prompt);
        model_.reset_kv();

        // Prefill in small chunks so the spinner can update.
        ui::Spinner spinner;
        spinner.start("thinking...");
        spinner.set_progress(0, static_cast<int>(ids.size()));

        const auto t0 = std::chrono::high_resolution_clock::now();
        const float* logits = nullptr;

        constexpr int kPrefillChunk = 32;
        for (size_t start = 0; start < ids.size(); start += kPrefillChunk) {
            const size_t end = std::min(start + kPrefillChunk, ids.size());
            std::vector<int> chunk(ids.begin() + start, ids.begin() + end);
            logits = model_.forward_batch(chunk);
            spinner.set_progress(static_cast<int>(end),
                static_cast<int>(ids.size()));
        }

        const auto t1 = std::chrono::high_resolution_clock::now();
        const double prefill_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        spinner.stop();

        // Generation loop.
        ui::ai_msg_begin();

        std::vector<int> generated;
        std::vector<int> recent = ids;
        int pos = static_cast<int>(ids.size());

        const auto g0 = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < max_tokens; ++step) {
            const int next = sampler_.sample(logits, model_.VOCAB, cfg_, recent);
            if (next == tok_id::kImEnd || next == tok_id::kEndOfText) break;

            generated.push_back(next);
            recent.push_back(next);

            ui::ai_msg_chunk(tok_.decode({ next }));
            logits = model_.forward(next, pos++);
        }

        const auto g1 = std::chrono::high_resolution_clock::now();
        const double gen_ms =
            std::chrono::duration<double, std::milli>(g1 - g0).count();
        const double tps = generated.empty()
            ? 0.0
            : generated.size() * 1000.0 / gen_ms;

        ui::ai_msg_end();

        char stat[192];
        std::snprintf(stat, sizeof(stat),
            "prefill %zu tok | %.0f ms   |   gen %zu tok | %.0f ms | %.1f tok/s",
            ids.size(), prefill_ms, generated.size(), gen_ms, tps);
        ui::info(stat);

        // Append this turn to history and cap the rolling window.
        history_.push_back({ "user", user_input });
        history_.push_back({ "assistant", tok_.decode(generated) });
        while (history_.size() > 6) history_.erase(history_.begin());

        return tok_.decode(generated);
    }

private:
    Qwen2Model& model_;
    Qwen2Tokenizer& tok_;
    Sampler sampler_;
    SamplerConfig cfg_;
    std::vector<ChatTurn> history_;
};

// ??? Interactive REPL ????????????????????????????????????????????
inline void run_agent(Qwen2Model& model, Qwen2Tokenizer& tok) {
    // The assistant is instructed to mirror the user's language, so
    // Chinese prompts get Chinese answers without any extra plumbing.
    const std::string kSystemPrompt =
        "You are a helpful, concise AI assistant. "
        "Always answer in the same language the user writes in: "
        "if the user writes in Chinese, reply in Chinese; "
        "if in English, reply in English; etc.";

    QwenAgent agent(model, tok);
    SamplerConfig cfg;
    agent.set_config(cfg);

    ui::banner();
    ui::hint_line();

    std::string line;
    while (true) {
        // Print the prompt, then read the user's line. The terminal echoes
        // the typed text itself, so no separate user_msg print is needed.
        ui::prompt();
        if (!ui::read_user_line(line)) break;

        // Trim surrounding whitespace.
        const size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        const size_t last = line.find_last_not_of(" \t\r\n");
        line = line.substr(first, last - first + 1);

        if (line == "/exit" || line == "/quit") {
            std::printf("\n  %sbye.%s\n\n",
                ui::color::gray(), ui::color::reset());
            break;
        }
        if (line == "/reset") {
            agent.clear_history();
            ui::notice("history cleared");
            continue;
        }
        if (line == "/clear") {
#ifdef _WIN32
            std::system("cls");
#else
            std::system("clear");
#endif
            ui::banner();
            continue;
        }
        if (line == "/help") {
            ui::info("/reset          clear conversation history");
            ui::info("/clear          clear screen");
            ui::info("/temp <n>       set temperature (0.1 - 2.0)");
            ui::info("/exit           quit");
            ui::info("Tip: you can type prompts in any language, including ??.");
            continue;
        }
        if (line.rfind("/temp ", 0) == 0) {
            try {
                const float t = std::stof(line.substr(6));
                cfg.temperature = t;
                agent.set_config(cfg);
                char b[64];
                std::snprintf(b, sizeof(b), "temperature = %.2f", t);
                ui::notice(b);
            }
            catch (...) {
                ui::error("usage: /temp 0.7");
            }
            continue;
        }
        if (!line.empty() && line[0] == '/') {
            ui::error("unknown command: " + line);
            continue;
        }

        agent.generate(kSystemPrompt, line, 1024);
    }
}