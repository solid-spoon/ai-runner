// cpp/agent.h — chat agent for Qwen2 / Qwen3 / MiniMind-style models.
#pragma once

#include "model.h"
#include "tokenizer.h"
#include "sampler.h"
#include "ui.h"
#include "config.h"

#include <cstdio>
#include <string>
#include <vector>
#include <chrono>
#include <sstream>
#include <algorithm>

struct ChatTurn {
    std::string role;
    std::string text;
};

class QwenAgent {
public:
    QwenAgent(QwenModel& model, Qwen2Tokenizer& tok, const ModelConfig& cfg)
        : model_(model), tok_(tok), cfg_(cfg) {
        eos_id_ = (cfg_.eos_token_id >= 0)
            ? cfg_.eos_token_id
            : (cfg_.vocab_size > 100000 ? 151645 : 2);

        if (cfg_.is_qwen3()) {
            think_open_id_ = 151667;
            think_close_id_ = 151668;
        }
        else if (cfg_.is_minimind()) {
            think_open_id_ = 25;
            think_close_id_ = 26;
        }
    }

    void set_config(const SamplerConfig& c) { cfg_s_ = c; }
    const SamplerConfig& config() const { return cfg_s_; }

    void enable_thinking() { enable_thinking_ = true; }
    void disable_thinking() { enable_thinking_ = false; }
    void toggle_thinking() { enable_thinking_ = !enable_thinking_; }
    bool thinking_enabled() const { return enable_thinking_; }
    bool supports_thinking() const { return think_open_id_ >= 0; }

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

        if (supports_thinking() && !enable_thinking_) {
            s << "<think>\n\n</think>\n\n";
        }
        return s.str();
    }

    std::string generate(const std::string& system_prompt,
        const std::string& user_input,
        int max_tokens = 1024) {

        const std::string prompt = build_prompt(system_prompt, user_input);
        const std::vector<int> ids = tok_.encode(prompt);

        size_t prefix = 0;
        const size_t min_len = std::min(cached_ids_.size(), ids.size());
        while (prefix < min_len && cached_ids_[prefix] == ids[prefix]) ++prefix;

        if (cached_ids_.size() > prefix || prefix >= ids.size()) {
            model_.reset_kv();
            cached_ids_.clear();
            prefix = 0;
        }
        const int n_cached = (int)prefix;
        const int n_new = (int)ids.size() - (int)prefix;

        ui::Spinner spinner;
        if (n_cached > 0) {
            char lbl[64];
            std::snprintf(lbl, sizeof(lbl), "prefilling (cached %d)...", n_cached);
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
            const size_t end = std::min(start + (size_t)kPrefillChunk, ids.size());
            std::vector<int> chunk(ids.begin() + start, ids.begin() + end);
            logits = model_.forward_batch(chunk);
            spinner.set_progress((int)(end - prefix), n_new);
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double prefill_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        spinner.stop();

        std::vector<int> generated;
        std::vector<int> recent = ids;
        int pos = (int)ids.size();

        bool in_think = false;
        bool prefix_printed = false;
        bool skip_leading_ws = false;

        const auto g0 = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < max_tokens; ++step) {
            const int next = sampler_.sample(logits, model_.VOCAB, cfg_s_, recent);
            if (next == eos_id_) break;

            generated.push_back(next);
            recent.push_back(next);

            if (next == think_open_id_) {
                in_think = true;
                ui::ai_msg_think_begin();
            }
            else if (next == think_close_id_) {
                in_think = false;
                ui::ai_msg_think_end();
                skip_leading_ws = true;
            }
            else if (in_think) {
                ui::ai_msg_think_chunk(tok_.decode({ next }));
            }
            else {
                std::string piece = tok_.decode({ next });
                if (skip_leading_ws) {
                    size_t i = 0;
                    while (i < piece.size() &&
                        (piece[i] == '\n' || piece[i] == '\r' ||
                            piece[i] == ' ' || piece[i] == '\t'))
                        ++i;
                    piece = piece.substr(i);
                    if (!piece.empty()) skip_leading_ws = false;
                }
                if (!piece.empty()) {
                    if (!prefix_printed) {
                        ui::ai_msg_begin();
                        prefix_printed = true;
                    }
                    ui::ai_msg_chunk(piece);
                }
            }
            logits = model_.forward(next, pos++);
        }
        if (!prefix_printed && !in_think) ui::ai_msg_begin();
        const auto g1 = std::chrono::high_resolution_clock::now();
        const double gen_ms =
            std::chrono::duration<double, std::milli>(g1 - g0).count();
        const double tps = generated.empty() ? 0.0
            : generated.size() * 1000.0 / gen_ms;
        ui::ai_msg_end();

        char stat[256];
        if (n_cached > 0) {
            std::snprintf(stat, sizeof(stat),
                "prefill %d new (cached %d) | %.0f ms   |   gen %zu tok | %.0f ms | %.1f tok/s",
                n_new, n_cached, prefill_ms, generated.size(), gen_ms, tps);
        }
        else {
            std::snprintf(stat, sizeof(stat),
                "prefill %d tok | %.0f ms   |   gen %zu tok | %.0f ms | %.1f tok/s",
                n_new, prefill_ms, generated.size(), gen_ms, tps);
        }
        ui::info(stat);

        cached_ids_ = ids;
        cached_ids_.insert(cached_ids_.end(), generated.begin(), generated.end());

        history_.push_back({ "user", user_input });
        history_.push_back({ "assistant", tok_.decode(generated) });
        while (history_.size() > 6) history_.erase(history_.begin());

        return tok_.decode(generated);
    }

private:
    QwenModel& model_;
    Qwen2Tokenizer& tok_;
    ModelConfig cfg_;
    Sampler sampler_;
    SamplerConfig cfg_s_;
    std::vector<ChatTurn> history_;
    std::vector<int> cached_ids_;
    int eos_id_ = 151645;
    bool enable_thinking_ = false;
    int  think_open_id_ = -1;
    int  think_close_id_ = -1;
};

inline void run_agent(QwenModel& model, Qwen2Tokenizer& tok,
    const ModelConfig& cfg) {
    const std::string kSystemPrompt =
        "You are a helpful, concise AI assistant. "
        "Always answer in the same language the user writes in.";

    QwenAgent agent(model, tok, cfg);
    SamplerConfig sc;

    if (cfg.is_qwen3()) {
        sc.temperature = 0.7f;
        sc.top_k = 20;
        sc.top_p = 0.8f;
        sc.repetition_penalty = 1.05f;
    }
    agent.set_config(sc);

    std::string title = cfg.model_type;
    if (cfg.model_type == "qwen3") title = "Qwen3";
    else if (cfg.model_type == "qwen2") title = "Qwen2.5";
    else if (cfg.model_type == "minimind") title = "MiniMind";

    ui::banner(title);
    ui::hint_line();

    if (agent.supports_thinking())
        ui::info("this model supports /think — toggle reasoning mode");

    std::string line;
    while (true) {
        ui::prompt();
        if (!ui::read_user_line(line)) break;

        const size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        const size_t last = line.find_last_not_of(" \t\r\n");
        line = line.substr(first, last - first + 1);

        if (line == "/exit" || line == "/quit") {
            std::printf("\n  %sbye.%s\n\n", ui::color::gray(), ui::color::reset());
            break;
        }
        if (line == "/reset") {
            agent.clear_history();
            ui::notice("history + KV cache cleared");
            continue;
        }
        if (line == "/clear") {
#ifdef _WIN32
            (void)std::system("cls");
#else
            (void)std::system("clear");
#endif
            ui::banner(title);
            continue;
        }
        if (line == "/think") {
            if (!agent.supports_thinking()) {
                ui::error("this model does not support thinking mode");
                continue;
            }
            agent.toggle_thinking();
            if (agent.thinking_enabled()) {
                ui::notice("thinking ON — the model will reason before answering");
            }
            else {
                ui::notice("thinking OFF — direct answers only");
            }
            continue;
        }
        if (line == "/help") {
            ui::info("/reset          clear conversation history + KV cache");
            ui::info("/clear          clear screen");
            ui::info("/think          toggle reasoning mode (Qwen3 / MiniMind)");
            ui::info("/temp <n>       set temperature (0.1 - 2.0)");
            ui::info("/exit           quit");
            continue;
        }
        if (line.rfind("/temp ", 0) == 0) {
            try {
                const float t = std::stof(line.substr(6));
                sc.temperature = t;
                agent.set_config(sc);
                char b[64]; std::snprintf(b, sizeof(b), "temperature = %.2f", t);
                ui::notice(b);
            }
            catch (...) { ui::error("usage: /temp 0.7"); }
            continue;
        }
        if (!line.empty() && line[0] == '/') {
            ui::error("unknown command: " + line);
            continue;
        }

        const int max_tok = agent.thinking_enabled() ? 4096 : 1024;
        agent.generate(kSystemPrompt, line, max_tok);
    }
}