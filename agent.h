#pragma once
#include "model.h"
#include "tokenizer.h"
#include "sampler.h"
#include "ui.h"
#include "config.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

struct ChatTurn { std::string role, text; };

class QwenAgent {
public:
    QwenAgent(QwenModel& m, Qwen2Tokenizer& t, const ModelConfig& c)
        : model_(m), tok_(t), cfg_(c)
    {
        eos_id_ = (cfg_.eos_token_id >= 0) ? cfg_.eos_token_id
            : (cfg_.vocab_size > 100000 ? 151645 : 2);
        // Дискриминатор — vocab_size, а не model_type.
        // MiniMind-3 в config.json пишет model_type="qwen3", но имеет
        // vocab=6400 и свои special-id. Если ориентироваться на
        // model_type, получим id вроде 151667, которых нет в embedding,
        // и OOB при чтении весов.
        if (cfg_.vocab_size <= 100000) {
            think_open_ = 25; think_close_ = 26;
        }
        else if (cfg_.is_qwen3()) {
            think_open_ = 151667; think_close_ = 151668;
        }
    }

    void set_config(const SamplerConfig& c) { cfg_s_ = c; }
    void set_temperature(float t) { cfg_s_.temperature = t; }
    float temperature() const { return cfg_s_.temperature; }
    void toggle_thinking() { enable_think_ = !enable_think_; }
    bool thinking() const { return enable_think_; }

    void clear() {
        history_.clear();
        cached_ids_.clear();
        model_.reset_kv();
    }

    std::string generate(const std::string& sys_prompt,
        const std::string& user_input,
        int max_tokens = 1024)
    {
        // 1. Промпт
        std::ostringstream ss;
        ss << "<|im_start|>system\n" << sys_prompt << "<|im_end|>\n";
        for (auto& t : history_)
            ss << "<|im_start|>" << t.role << "\n" << t.text << "<|im_end|>\n";
        ss << "<|im_start|>user\n" << user_input << "<|im_end|>\n"
            << "<|im_start|>assistant\n";
        if (think_open_ >= 0 && !enable_think_)
            ss << "<think>\n\n</think>\n\n";

        std::vector<int> ids = tok_.encode(ss.str());
        if (ids.empty()) return "";

        // 2. Учитываем max_tokens: иначе forward() бросит во время декода,
        // когда kv_len_ доползёт до kMaxContext посреди генерации.
        if (model_.kv_len() + (int)ids.size() + max_tokens > kMaxContext) {
            model_.reset_kv();
            cached_ids_.clear();
        }

        // 3. Префикс кэш
        size_t prefix = 0;
        size_t mn = std::min(cached_ids_.size(), ids.size());
        while (prefix < mn && cached_ids_[prefix] == ids[prefix]) prefix++;

        if (cached_ids_.size() > prefix) {
            model_.reset_kv();
            cached_ids_.clear();
            prefix = 0;
        }

        // 4. Префилл.
        const float* logits = nullptr;
        for (size_t i = prefix; i < ids.size(); i += 32) {
            size_t end = std::min(i + 32, ids.size());
            std::vector<int> chunk(ids.begin() + i, ids.begin() + end);
            logits = model_.forward_batch(chunk);
        }

        // Фикс: если логитс пуст (весь промпт был в кэше), форсируем пересчёт.
        if (!logits) {
            model_.reset_kv();
            cached_ids_.clear();
            logits = model_.forward_batch(ids);
        }

        // 5. Генерация
        ui::ai_msg_begin();
        std::vector<int> out;
        std::vector<int> recent = ids;
        bool in_think = false;

        for (int step = 0; step < max_tokens; step++) {
            int next = sampler_.sample(logits, model_.VOCAB, cfg_s_, recent);
            if (next == eos_id_) break;
            out.push_back(next);
            recent.push_back(next);

            if (next == think_open_) {
                in_think = true;
                ui::ai_msg_think_begin();
            }
            else if (next == think_close_) {
                in_think = false;
                ui::ai_msg_think_end();
            }
            else {
                std::string p = tok_.decode({ next });
                if (!p.empty()) {
                    if (in_think) ui::ai_msg_think_chunk(p);
                    else ui::ai_msg_chunk(p);
                }
            }
            logits = model_.forward(next);
        }
        ui::ai_msg_end();

        // 6. Сохраняем состояние
        std::string reply = tok_.decode(out);
        history_.push_back({ "user", user_input });
        history_.push_back({ "assistant", reply });
        while (history_.size() > 6) history_.erase(history_.begin());

        cached_ids_ = ids;
        cached_ids_.insert(cached_ids_.end(), out.begin(), out.end());

        return reply;
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
    int think_open_ = -1, think_close_ = -1;
    bool enable_think_ = false;
};

// ─── REPL ─────────────────────────────────────────────────────────────
inline int run_agent(QwenModel& model, Qwen2Tokenizer& tok, const ModelConfig& cfg) {
    ui::init();
    std::string name = cfg.is_qwen3() ? "Qwen3" : cfg.is_qwen2() ? "Qwen2.5" : "model";
    ui::banner(name);

    QwenAgent agent(model, tok, cfg);
    if (cfg.is_qwen3()) {
        SamplerConfig sc;
        sc.temperature = 0.7f; sc.top_k = 20; sc.top_p = 0.8f;
        sc.repetition_penalty = 1.05f;
        agent.set_config(sc);
    }

    const char* sys = "You are a helpful AI assistant. Answer in the user's language.";

    for (;;) {
        ui::prompt();
        std::string line;
        if (!ui::read_user_line(line)) break;
        if (line.empty()) continue;

        if (line == "/exit") break;
        if (line == "/reset") { agent.clear(); ui::notice("cleared"); continue; }
        if (line == "/help") { ui::help(); continue; }
        if (line == "/clear") {
#ifdef _WIN32
            std::system("cls");
#else
            std::system("clear");
#endif
            ui::banner(name);
            continue;
        }
        if (line == "/think") {
            agent.toggle_thinking();
            ui::notice(agent.thinking() ? "thinking ON" : "thinking OFF");
            continue;
        }
        if (line.rfind("/temp", 0) == 0) {
            try {
                float t = std::stof(line.substr(5));
                agent.set_temperature(t);
                char buf[64];
                std::snprintf(buf, sizeof(buf), "temperature = %.2f", t);
                ui::notice(buf);
            }
            catch (...) { ui::error("usage: /temp <num>"); }
            continue;
        }

        try { agent.generate(sys, line); }
        catch (const std::exception& e) { ui::error(e.what()); }
    }
    return 0;
}