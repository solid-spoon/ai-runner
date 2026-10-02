// cpp/agent.h — Qwen2 chat agent driving the TUI, with prefix caching.
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
#include <algorithm>

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

    // Clears both the dialogue history AND the KV cache tracker, so the
    // next generate() call falls back to a full prefill.
    void clear_history() {
        history_.clear();
        cached_ids_.clear();
        model_.reset_kv();
    }

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

    std::string generate(const std::string& system_prompt,
        const std::string& user_input,
        int max_tokens = 1024) {

        const std::string prompt = build_prompt(system_prompt, user_input);
        const std::vector<int> ids = tok_.encode(prompt);

        // ??? 1. PREFIX MATCH ???????????????????????????????????????
        // How many leading tokens of the new prompt are already sitting
        // in the KV cache?
        size_t prefix = 0;
        const size_t min_len = std::min(cached_ids_.size(), ids.size());
        while (prefix < min_len && cached_ids_[prefix] == ids[prefix]) {
            ++prefix;
        }

        // We can't truncate the KV cache in place, and we always need at
        // least one fresh token so the final logits are valid. If either
        // condition is violated, nuke everything and start over.
        if (cached_ids_.size() > prefix || prefix >= ids.size()) {
            model_.reset_kv();
            cached_ids_.clear();
            prefix = 0;
        }

        const int n_cached = static_cast<int>(prefix);
        const int n_new = static_cast<int>(ids.size() - prefix);

        // ??? 2. PREFILL (suffix only) ???????????????????????????????
        ui::Spinner spinner;
        if (n_cached > 0) {
            char lbl[64];
            std::snprintf(lbl, sizeof(lbl),
                "prefilling (cached %d)...", n_cached);
            spinner.start(lbl);
        }
        else {
            spinner.start("thinking...");
        }
        spinner.set_progress(0, n_new);

        const auto t0 = std::chrono::high_resolution_clock::now();
        const float* logits = nullptr;

        constexpr int kPrefillChunk = 32;
        for (size_t start = prefix; start < ids.size(); start += kPrefillChunk) {
            const size_t end = std::min(start + kPrefillChunk, ids.size());
            std::vector<int> chunk(ids.begin() + start, ids.begin() + end);
            logits = model_.forward_batch(chunk);
            spinner.set_progress(static_cast<int>(end - prefix), n_new);
        }

        const auto t1 = std::chrono::high_resolution_clock::now();
        const double prefill_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        spinner.stop();

        // ??? 3. GENERATION ??????????????????????????????????????????
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

        // ??? 4. STATS ???????????????????????????????????????????????
        char stat[256];
        if (n_cached > 0) {
            std::snprintf(stat, sizeof(stat),
                "prefill %d new (cached %d) | %.0f ms   |   gen %zu tok | %.0f ms | %.1f tok/s",
                n_new, n_cached, prefill_ms,
                generated.size(), gen_ms, tps);
        }
        else {
            std::snprintf(stat, sizeof(stat),
                "prefill %d tok | %.0f ms   |   gen %zu tok | %.0f ms | %.1f tok/s",
                n_new, prefill_ms,
                generated.size(), gen_ms, tps);
        }
        ui::info(stat);

        // ??? 5. TRACK WHAT'S NOW IN THE KV CACHE ???????????????????
        // KV holds: every prompt token + every generated token.
        // (The stopping token is sampled but never forwarded, so it's
        //  not part of the cache — matching the next prompt's prefix
        //  logic, which sees <|im_end|> as the first "new" token.)
        cached_ids_ = ids;
        cached_ids_.insert(cached_ids_.end(),
            generated.begin(), generated.end());

        // ??? 6. UPDATE HISTORY ??????????????????????????????????????
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
    std::vector<int> cached_ids_;
};

// ??? Interactive REPL ???????????????????????????????????????????
inline void run_agent(Qwen2Model& model, Qwen2Tokenizer& tok) {
    const std::string kSystemPrompt =
        "You are a helpful, concise AI assistant. "
        "Always answer in the same language the user writes in.";

    QwenAgent agent(model, tok);
    SamplerConfig cfg;
    agent.set_config(cfg);

    ui::banner();
    ui::hint_line();

    std::string line;
    while (true) {
        ui::prompt();
        if (!ui::read_user_line(line)) break;

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
            ui::notice("history + KV cache cleared");
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
            ui::info("/reset          clear conversation history + KV cache");
            ui::info("/clear          clear screen");
            ui::info("/temp <n>       set temperature (0.1 - 2.0)");
            ui::info("/exit           quit");
            ui::info("Tip: follow-up questions reuse the KV cache and are much faster.");
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