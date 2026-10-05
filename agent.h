#pragma once
#include "model.h"
#include "tokenizer.h"
#include "sampler.h"
#include "ui.h"
#include "config.h"
#include "tools.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

struct ChatTurn { std::string role, text; };

inline const char* kSystemAgentZH =
"你是一个自主的文件与编码代理。你在沙箱（当前工作目录）中工作。\n"
"通过调用工具来完成任务，不要用自然语言描述工具调用。\n\n"

"规则：\n"
"1. 用用户的语言回复。\n"
"2. 修改文件前先用 read_file 读取；路径未知时先用 list_dir 查找。\n"
"3. 优先做局部修改（append_file 或精确重写）。除非有明确理由，不要整体重写文件。\n"
"4. 每条消息只输出一个工具调用，前后不要有任何文字：\n"
"<tool_call>\n"
"{\"name\": \"<tool>\", \"arguments\": {\"<arg>\": \"<value>\"}}\n"
"</tool_call>\n"
"5. 路径必须是相对路径，且位于沙箱内。绝对路径会被拒绝。\n"
"6. 收到 <tool_response> 后直接使用结果，不要复述文件内容。\n"
"7. 任务完成后停止调用工具，给出简短的最终总结。\n"
"8. 不要编造文件内容、路径或名称。不确定时，读取、列目录或询问。\n\n"

"示例：\n"
"user: 看一下 src/main.py 里有什么\n"
"assistant:\n"
"<tool_call>\n"
"{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/main.py\"}}\n"
"</tool_call>\n\n"

"最终总结格式（仅在最后一次工具调用之后；字段名保持英文原样，内容用用户语言）：\n"
"Files: <涉及的文件路径>\n"
"Run:   <一条运行命令，若不适用填 '—'>\n"
"Notes: <1-2 行，仅在需要时>\n\n"

"边界情况：\n"
"- 工具报错 → 修正参数重试（每步最多 2 次）。\n"
"- 结果为空 → 目标不存在，不要重复同样的调用。\n"
"- 缺少必要工具 → 直接说明，不要猜测。\n"
"- 语法或依赖不清 → 用一句话提问。\n\n";

inline const char* kSystemChatZH =
"你是一个友好的助手。简短、直接地回复，使用用户的语言。\n\n"
"规则：\n"
"1. 不知道就直说“不知道”。不要编造事实、数字、姓名、日期。\n"
"2. 不要前缀（“当然！”、“好问题！”）。直接回答。\n"
"3. 不要重复用户的问题。\n"
"4. 示例只在被要求时给出。\n\n"
"边界情况：\n"
"- 输入无意义 → 请用户重新表述。\n"
"- 超出知识范围 → 明确说明。\n";

inline std::string build_system(const tools::Config& tc) {
    if (!tc.enabled) return std::string(kSystemChatZH);
    return std::string(kSystemAgentZH) + tools::describe(tc);
}

class QwenAgent {
public:
    QwenAgent(QwenModel& m, Qwen2Tokenizer& t, const ModelConfig& c)
        : model_(m), tok_(t), cfg_(c)
    {
        eos_id_ = (cfg_.eos_token_id >= 0) ? cfg_.eos_token_id
            : (cfg_.vocab_size > 100000 ? 151645 : 2);
        if (cfg_.vocab_size <= 100000) { think_open_ = 25; think_close_ = 26; }
        else if (cfg_.is_qwen3()) { think_open_ = 151667; think_close_ = 151668; }
    }

    void set_config(const SamplerConfig& c) { cfg_s_ = c; }
    void set_temperature(float t) { cfg_s_.temperature = t; }
    float temperature() const { return cfg_s_.temperature; }
    void toggle_thinking() { enable_think_ = !enable_think_; }
    bool thinking() const { return enable_think_; }
    void set_tools_config(const tools::Config& c) { tools_cfg_ = c; }
    tools::Config& tools_config() { return tools_cfg_; }

    void clear() {
        history_.clear(); cached_ids_.clear(); model_.reset_kv();
    }

    std::string generate(const std::string& sys_prompt,
        const std::string& user_input,
        int max_tokens = 1024)
    {
        history_.push_back({ "user", user_input });
        std::string final_reply;

        const int rounds = tools_cfg_.enabled ? tools_cfg_.max_rounds : 1;
        for (int round = 0; round < rounds; ++round) {
            std::string reply = generate_one_round(sys_prompt, max_tokens);
            final_reply = reply;

            if (!tools_cfg_.enabled) break;
            tools::Call call;
            if (!tools::find_tool_call(reply, call)) break;

            tools::Result r = tools::execute(tools_cfg_, call);
            ui::tool_result(call.name, r.ok, r.text);

            std::string resp = "<tool_response>\n" + r.text + "\n</tool_response>";
            history_.push_back({ "tool", resp });
        }

        trim_history();
        return final_reply;
    }

private:
    void trim_history() {
        while (history_.size() > 12) {
            history_.erase(history_.begin());
            if (!history_.empty() && history_.front().role == "tool")
                history_.erase(history_.begin());
        }
    }

    std::string generate_one_round(const std::string& sys_prompt, int max_tokens) {
        std::ostringstream ss;
        ss << "<|im_start|>system\n" << sys_prompt << "<|im_end|>\n";
        for (auto& t : history_)
            ss << "<|im_start|>" << t.role << "\n" << t.text << "<|im_end|>\n";
        ss << "<|im_start|>assistant\n";
        if (think_open_ >= 0 && !enable_think_)
            ss << "<think>\n\n</think>\n\n";

        std::vector<int> ids = tok_.encode(ss.str());
        if (ids.empty()) return "";

        if (model_.kv_len() + (int)ids.size() + max_tokens > kMaxContext) {
            model_.reset_kv(); cached_ids_.clear();
        }

        size_t prefix = 0;
        size_t mn = std::min(cached_ids_.size(), ids.size());
        while (prefix < mn && cached_ids_[prefix] == ids[prefix]) prefix++;
        if (cached_ids_.size() > prefix) { model_.reset_kv(); cached_ids_.clear(); prefix = 0; }

        const float* logits = nullptr;
        for (size_t i = prefix; i < ids.size(); i += 32) {
            size_t end = std::min(i + 32, ids.size());
            std::vector<int> chunk(ids.begin() + i, ids.begin() + end);
            logits = model_.forward_batch(chunk);
        }
        if (!logits) {
            model_.reset_kv(); cached_ids_.clear();
            logits = model_.forward_batch(ids);
        }

        ui::ai_msg_begin();
        std::vector<int> out;
        std::vector<int> recent = ids;
        bool in_think = false;
        std::string full_msg;

        const std::string TC_OPEN = "<tool_call>";
        const std::string TC_CLOSE = "</tool_call>";
        std::string scan_buf;
        std::string tool_body;
        bool in_tool_block = false;

        auto send_text = [&](const std::string& s) {
            if (s.empty()) return;
            if (in_think) ui::ai_msg_think_chunk(s);
            else          ui::ai_msg_chunk(s);
        };

        auto feed_tool_scan = [&](const std::string& p) {
            if (in_tool_block) {
                tool_body += p;
                if (tool_body.size() >= TC_CLOSE.size() &&
                    tool_body.compare(tool_body.size() - TC_CLOSE.size(),
                        TC_CLOSE.size(), TC_CLOSE) == 0) {
                    std::string body = tool_body.substr(0, tool_body.size() - TC_CLOSE.size());
                    ui::tool_call_marker(body);
                    in_tool_block = false;
                    tool_body.clear();
                }
                return;
            }
            scan_buf += p;
            size_t found = scan_buf.find(TC_OPEN);
            if (found != std::string::npos) {
                send_text(scan_buf.substr(0, found));
                in_tool_block = true;
                tool_body = scan_buf.substr(found + TC_OPEN.size());
                scan_buf.clear();
                return;
            }
            size_t keep = std::min(TC_OPEN.size() - 1, scan_buf.size());
            if (scan_buf.size() > keep) {
                size_t n = scan_buf.size() - keep;
                send_text(scan_buf.substr(0, n));
                scan_buf = scan_buf.substr(n);
            }
        };

        for (int step = 0; step < max_tokens; step++) {
            int next = sampler_.sample(logits, model_.VOCAB, cfg_s_, recent);
            if (next == eos_id_) break;
            out.push_back(next);
            recent.push_back(next);

            std::string p = tok_.decode({ next });
            full_msg += p;

            if (next == think_open_) {
                in_think = true;
                ui::ai_msg_think_begin();
            }
            else if (next == think_close_) {
                in_think = false;
                ui::ai_msg_think_end();
            }
            else if (in_think) {
                ui::ai_msg_think_chunk(p);
            }
            else {
                feed_tool_scan(p);
            }
            logits = model_.forward(next);
        }

        if (!scan_buf.empty() && !in_tool_block) {
            send_text(scan_buf);
            scan_buf.clear();
        }
        if (in_tool_block) {
            ui::tool_call_marker(tool_body);
            in_tool_block = false;
        }
        ui::ai_msg_end();

        history_.push_back({ "assistant", full_msg });
        cached_ids_ = ids;
        cached_ids_.insert(cached_ids_.end(), out.begin(), out.end());
        return full_msg;
    }

    QwenModel& model_;
    Qwen2Tokenizer& tok_;
    ModelConfig cfg_;
    Sampler sampler_;
    SamplerConfig cfg_s_;
    tools::Config tools_cfg_;
    std::vector<ChatTurn> history_;
    std::vector<int> cached_ids_;
    int eos_id_ = 151645;
    int think_open_ = -1, think_close_ = -1;
    bool enable_think_ = false;
};

inline void apply_agent_sampler(QwenAgent& agent) {
    SamplerConfig sc;
    sc.temperature = 0.2f;
    sc.top_p = 0.9f;
    sc.top_k = 20;
    sc.repetition_penalty = 1.05f;
    agent.set_config(sc);
}

inline void apply_chat_sampler(QwenAgent& agent) {
    SamplerConfig sc;
    sc.temperature = 0.6f;
    sc.top_p = 0.9f;
    sc.top_k = 40;
    sc.repetition_penalty = 1.05f;
    agent.set_config(sc);
}

inline int run_agent(QwenModel& model, Qwen2Tokenizer& tok, const ModelConfig& cfg) {
    ui::init();
    std::string name = cfg.is_qwen3() ? "Qwen3" : cfg.is_qwen2() ? "Qwen2.5" : "model";
    ui::banner(name);

    QwenAgent agent(model, tok, cfg);

    tools::Config tcfg;
    tcfg.enabled = true;
    tcfg.allow_write = true;
    tcfg.allow_delete = false;
    agent.set_tools_config(tcfg);

    // Agent mode by default: low temperature for reliable tool-call format.
    apply_agent_sampler(agent);

    std::string sys = build_system(tcfg);

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
        if (line == "/tools") {
            auto& tc = agent.tools_config();
            tc.enabled = !tc.enabled;
            ui::notice(tc.enabled ? "tools ON" : "tools OFF");
            sys = build_system(tc);
            if (tc.enabled) apply_agent_sampler(agent);
            else            apply_chat_sampler(agent);
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

        try {
            const int max_new = tcfg.enabled ? 512 : 1024;
            agent.generate(sys, line, max_new);
        }
        catch (const std::exception& e) { ui::error(e.what()); }
    }
    return 0;
}
