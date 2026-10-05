#pragma once
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <filesystem>

namespace tools {

    struct Config {
        std::string root;
        bool        enabled = true;
        bool        allow_write = true;
        bool        allow_delete = false;
        size_t      max_read_bytes = 64 * 1024;
        int         max_rounds = 6;
    };

    struct Result { bool ok = false; std::string text; };
    struct Call { std::string name; std::string args_json; };

    namespace json {

        inline void skip_ws(const std::string& s, size_t& i) {
            while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
        }

        struct Value {
            enum T { Nul, Str, Obj, Arr, Num } t = Nul;
            std::string s;
            std::vector<std::pair<std::string, Value>> o;
        };

        inline bool parse_string(const std::string& s, size_t& i, std::string& out) {
            if (i >= s.size() || s[i] != '"') return false;
            ++i; out.clear();
            while (i < s.size() && s[i] != '"') {
                if (s[i] != '\\') { out += s[i++]; continue; }
                ++i; if (i >= s.size()) return false;
                switch (s[i]) {
                case 'n': out += '\n'; ++i; break;
                case 't': out += '\t'; ++i; break;
                case 'r': out += '\r'; ++i; break;
                case 'b': out += '\b'; ++i; break;
                case 'f': out += '\f'; ++i; break;
                case '"': out += '"';  ++i; break;
                case '\\': out += '\\'; ++i; break;
                case '/': out += '/';  ++i; break;
                case 'u': {
                    if (i + 4 >= s.size()) return false;
                    uint32_t cp = 0;
                    for (int k = 1; k <= 4; ++k) {
                        const char h = s[i + k];
                        cp <<= 4;
                        if (h >= '0' && h <= '9')      cp |= (h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                        else return false;
                    }
                    i += 5;
                    if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 < s.size() &&
                        s[i] == '\\' && s[i + 1] == 'u') {
                        uint32_t cp2 = 0; bool ok = true;
                        for (int k = 2; k <= 5; ++k) {
                            const char h = s[i + k];
                            cp2 <<= 4;
                            if (h >= '0' && h <= '9')      cp2 |= (h - '0');
                            else if (h >= 'a' && h <= 'f') cp2 |= (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp2 |= (h - 'A' + 10);
                            else { ok = false; break; }
                        }
                        if (ok && cp2 >= 0xDC00 && cp2 <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (cp2 - 0xDC00);
                            i += 6;
                        }
                    }
                    if (cp < 0x80) out += static_cast<char>(cp);
                    else if (cp < 0x800) {
                        out += static_cast<char>(0xC0 | (cp >> 6));
                        out += static_cast<char>(0x80 | (cp & 0x3F));
                    }
                    else if (cp < 0x10000) {
                        out += static_cast<char>(0xE0 | (cp >> 12));
                        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                        out += static_cast<char>(0x80 | (cp & 0x3F));
                    }
                    else {
                        out += static_cast<char>(0xF0 | (cp >> 18));
                        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                        out += static_cast<char>(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: out += s[i++]; break;
                }
            }
            if (i >= s.size()) return false;
            ++i; return true;
        }

        inline bool parse_value(const std::string& s, size_t& i, Value& out, int depth);
        inline bool parse_object(const std::string& s, size_t& i, Value& out, int depth);

        inline bool parse_value(const std::string& s, size_t& i, Value& out, int depth) {
            if (depth > 16) return false;
            skip_ws(s, i);
            if (i >= s.size()) return false;
            const char c = s[i];
            if (c == '"') { out.t = Value::Str; return parse_string(s, i, out.s); }
            if (c == '{') return parse_object(s, i, out, depth);
            if (c == '[') {
                out.t = Value::Arr;
                int d = 0; bool in_str = false;
                while (i < s.size()) {
                    char ch = s[i];
                    if (in_str) {
                        if (ch == '\\') { i += 2; continue; }
                        if (ch == '"') in_str = false;
                    }
                    else {
                        if (ch == '"') in_str = true;
                        else if (ch == '[') ++d;
                        else if (ch == ']') { --d; if (d == 0) { ++i; return true; } }
                    }
                    ++i;
                }
                return false;
            }
            out.t = Value::Num;
            const size_t start = i;
            while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' &&
                s[i] != ' ' && s[i] != '\t' && s[i] != '\n' && s[i] != '\r')
                ++i;
            out.s = s.substr(start, i - start);

            return true;
        }

        inline bool parse_object(const std::string& s, size_t& i, Value& out, int depth) {
            if (i >= s.size() || s[i] != '{') return false;
            ++i;
            out.t = Value::Obj;
            skip_ws(s, i);
            if (i < s.size() && s[i] == '}') { ++i; return true; }
            while (true) {
                skip_ws(s, i);
                std::string key;
                if (!parse_string(s, i, key)) return false;
                skip_ws(s, i);
                if (i >= s.size() || s[i] != ':') return false;
                ++i;
                Value v;
                if (!parse_value(s, i, v, depth + 1)) return false;
                out.o.emplace_back(std::move(key), std::move(v));
                skip_ws(s, i);
                if (i >= s.size()) return false;
                if (s[i] == ',') { ++i; continue; }
                if (s[i] == '}') { ++i; return true; }
                return false;
            }
        }

        inline const Value* find(const Value& o, const char* key) {
            if (o.t != Value::Obj) return nullptr;
            for (auto& kv : o.o) if (kv.first == key) return &kv.second;
            return nullptr;
        }

        inline std::string escape(const std::string& s) {
            std::string out; out.reserve(s.size() + 2); out += '"';
            for (unsigned char c : s) {
                switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                case '\b': out += "\\b";  break;
                case '\f': out += "\\f";  break;
                default:
                    if (c < 0x20) {
                        char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    }
                    else out += static_cast<char>(c);
                }
            }
            out += '"'; return out;
        }

    } // namespace json

    inline std::string safe_resolve(const Config& cfg, const std::string& rel) {
        namespace fs = std::filesystem;
        if (rel.empty()) throw std::runtime_error("empty path");
        fs::path root = cfg.root.empty() ? fs::current_path() : fs::path(cfg.root);
        root = fs::absolute(root);
        fs::path p(rel);
        if (p.is_absolute()) throw std::runtime_error("absolute paths not allowed");
        fs::path full = fs::weakly_canonical(root / p);
        const std::string rs = root.string();
        const std::string fs_str = full.string();
        if (fs_str.size() < rs.size() || fs_str.compare(0, rs.size(), rs) != 0)
            throw std::runtime_error("path escapes sandbox");
        if (fs_str.size() > rs.size() && fs_str[rs.size()] != '/' && fs_str[rs.size()] != '\\')
            throw std::runtime_error("path escapes sandbox");
        return fs_str;
    }

    inline Result read_file(const Config& cfg, const std::string& path) {
        try {
            const std::string full = safe_resolve(cfg, path);
            std::ifstream f(full, std::ios::binary);
            if (!f) return { false, "file not found: " + path };
            std::stringstream ss; char buf[8192]; size_t total = 0;
            while (f && total < cfg.max_read_bytes) {
                f.read(buf, sizeof(buf));
                const size_t got = static_cast<size_t>(f.gcount());
                if (!got) break;
                const size_t take = std::min(got, cfg.max_read_bytes - total);
                ss.write(buf, take); total += take;
            }
            std::string s = ss.str();
            if (total >= cfg.max_read_bytes)
                s += "\n[truncated at " + std::to_string(cfg.max_read_bytes) + " bytes]";
            return { true, std::move(s) };
        }
        catch (const std::exception& e) {
            return { false, std::string("error: ") + e.what() };
        }
    }

    inline Result write_file(const Config& cfg, const std::string& path,
        const std::string& content) {
        if (!cfg.allow_write) return { false, "write is disabled" };
        try {
            const std::string full = safe_resolve(cfg, path);
            std::filesystem::path fp(full);
            if (fp.has_parent_path()) std::filesystem::create_directories(fp.parent_path());
            std::ofstream f(full, std::ios::binary | std::ios::trunc);
            if (!f) return { false, "cannot open for write: " + path };
            f.write(content.data(), static_cast<std::streamsize>(content.size()));
            if (!f) return { false, "write failed" };
            return { true, "wrote " + std::to_string(content.size()) + " bytes to " + path };
        }
        catch (const std::exception& e) {
            return { false, std::string("error: ") + e.what() };
        }
    }

    inline Result append_file(const Config& cfg, const std::string& path,
        const std::string& content) {
        if (!cfg.allow_write) return { false, "write is disabled" };
        try {
            const std::string full = safe_resolve(cfg, path);
            std::filesystem::path fp(full);
            if (fp.has_parent_path()) std::filesystem::create_directories(fp.parent_path());
            std::ofstream f(full, std::ios::binary | std::ios::app);
            if (!f) return { false, "cannot open for append: " + path };
            f.write(content.data(), static_cast<std::streamsize>(content.size()));
            if (!f) return { false, "append failed" };
            return { true, "appended " + std::to_string(content.size()) + " bytes to " + path };
        }
        catch (const std::exception& e) {
            return { false, std::string("error: ") + e.what() };
        }
    }

    inline Result delete_file(const Config& cfg, const std::string& path) {
        if (!cfg.allow_delete) return { false, "delete is disabled" };
        try {
            const std::string full = safe_resolve(cfg, path);
            if (!std::filesystem::exists(full)) return { false, "file not found: " + path };
            std::filesystem::remove(full);
            return { true, "deleted " + path };
        }
        catch (const std::exception& e) {
            return { false, std::string("error: ") + e.what() };
        }
    }

    inline Result list_dir(const Config& cfg, const std::string& path) {
        try {
            const std::string p = path.empty() ? "." : path;
            const std::string full = safe_resolve(cfg, p);
            if (!std::filesystem::exists(full)) return { false, "not found: " + p };
            if (!std::filesystem::is_directory(full)) return { false, "not a directory: " + p };
            std::string out;
            for (auto& e : std::filesystem::directory_iterator(full)) {
                const auto name = e.path().filename().string();
                const bool is_dir = e.is_directory();
                out += is_dir ? "[dir]  " : "[file] ";
                out += name;
                if (!is_dir) {
                    std::error_code ec;
                    auto sz = std::filesystem::file_size(e.path(), ec);
                    if (!ec) out += "  (" + std::to_string(sz) + " B)";
                }
                out += '\n';
            }
            if (out.empty()) out = "(empty)";
            return { true, std::move(out) };
        }
        catch (const std::exception& e) {
            return { false, std::string("error: ") + e.what() };
        }
    }

    inline bool find_tool_call(const std::string& text, Call& out) {
        const std::string open_tag = "<tool_call>";
        const std::string close_tag = "</tool_call>";
        const size_t open = text.rfind(open_tag);
        if (open == std::string::npos) return false;
        const size_t body_start = open + open_tag.size();
        const size_t close = text.find(close_tag, body_start);
        std::string body = (close == std::string::npos)
            ? text.substr(body_start)
            : text.substr(body_start, close - body_start);
        size_t a = body.find_first_not_of(" \t\r\n");
        size_t b = body.find_last_not_of(" \t\r\n");
        if (a == std::string::npos) return false;
        body = body.substr(a, b - a + 1);

        json::Value root; size_t i = 0;
        if (!json::parse_value(body, i, root, 0)) return false;
        const json::Value* nm = json::find(root, "name");
        if (!nm || nm->t != json::Value::Str) return false;
        out.name = nm->s;

        const json::Value* args = json::find(root, "arguments");
        if (args && args->t == json::Value::Obj) {
            std::string s = "{";
            for (size_t k = 0; k < args->o.size(); ++k) {
                if (k) s += ",";
                s += json::escape(args->o[k].first) + ":";
                if (args->o[k].second.t == json::Value::Str)
                    s += json::escape(args->o[k].second.s);
                else
                    s += "\"\"";
            }
            s += "}";
            out.args_json = std::move(s);
        }
        else {
            out.args_json = "{}";
        }
        return true;
    }

    inline Result execute(const Config& cfg, const Call& call) {
        json::Value root; size_t i = 0;
        const json::Value* args = nullptr;
        if (json::parse_value(call.args_json, i, root, 0) && root.t == json::Value::Obj)
            args = &root;
        auto get = [&](const char* key) -> std::string {
            const json::Value* v = args ? json::find(*args, key) : nullptr;
            return (v && v->t == json::Value::Str) ? v->s : std::string();
        };

        if (call.name == "read_file")   return read_file(cfg, get("path"));
        if (call.name == "write_file")  return write_file(cfg, get("path"), get("content"));
        if (call.name == "append_file") return append_file(cfg, get("path"), get("content"));
        if (call.name == "delete_file") return delete_file(cfg, get("path"));
        if (call.name == "list_dir")    return list_dir(cfg, get("path"));
        return { false, "unknown tool: " + call.name };
    }

    inline std::string describe(const Config& cfg) {
        std::string s;
        s += "You have access to file tools.\n";
        s += "To call a tool, output exactly:\n";
        s += "<tool_call>\n";
        s += "{\"name\": \"<tool_name>\", \"arguments\": {\"<arg>\": \"<value>\"}}\n";
        s += "</tool_call>\n";
        s += "The runtime will execute it and reply with a <tool_response> block.\n";
        s += "\nAvailable tools:\n";
        s += "- read_file(path): read a file\n";
        if (cfg.allow_write) {
            s += "- write_file(path, content): create or overwrite\n";
            s += "- append_file(path, content): append\n";
        }
        if (cfg.allow_delete)
            s += "- delete_file(path): delete a file\n";
        s += "- list_dir(path): list entries\n";
        s += "\nRules:\n";
        s += "- Use relative paths only. The sandbox root is the working directory.\n";
        s += "- After a tool runs you will see a <tool_response>. Continue from there.\n";
        s += "- If a tool fails, explain the error to the user.\n";
        return s;
    }

} // namespace tools