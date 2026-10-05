#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cctype>
#include <climits>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <fstream>
#include <sstream>

inline std::string utf8_encode(uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    }
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
    return out;
}

inline std::pair<uint32_t, int> utf8_decode(const std::string& s, size_t i) {
    const uint8_t c = static_cast<uint8_t>(s[i]);
    if (c < 0x80) return { c, 1 };
    if ((c & 0xE0) == 0xC0 && i + 1 < s.size())
        return { (static_cast<uint32_t>(c & 0x1F) << 6) |
                 (static_cast<uint8_t>(s[i + 1]) & 0x3F), 2 };
    if ((c & 0xF0) == 0xE0 && i + 2 < s.size())
        return { (static_cast<uint32_t>(c & 0x0F) << 12) |
                 ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 6) |
                 (static_cast<uint8_t>(s[i + 2]) & 0x3F), 3 };
    if ((c & 0xF8) == 0xF0 && i + 3 < s.size())
        return { (static_cast<uint32_t>(c & 0x07) << 18) |
                 ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 12) |
                 ((static_cast<uint8_t>(s[i + 2]) & 0x3F) << 6) |
                 (static_cast<uint8_t>(s[i + 3]) & 0x3F), 4 };
    return { c, 1 };
}

inline bool is_letter_cp(uint32_t cp) {
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return true;
    return cp >= 0x80;
}
inline bool is_digit_cp(uint32_t cp) { return cp >= '0' && cp <= '9'; }
inline bool is_whitespace_cp(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
        cp == 0x0B || cp == 0x0C;
}

inline const std::vector<std::pair<uint8_t, uint32_t>>& byte_uni_table() {
    static std::vector<std::pair<uint8_t, uint32_t>> table;
    if (!table.empty()) return table;
    for (int b = 33; b <= 126; ++b) table.push_back({ (uint8_t)b, (uint32_t)b });
    for (int b = 161; b <= 172; ++b) table.push_back({ (uint8_t)b, (uint32_t)b });
    for (int b = 174; b <= 255; ++b) table.push_back({ (uint8_t)b, (uint32_t)b });
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (b >= 33 && b <= 126) continue;
        if (b >= 161 && b <= 172) continue;
        if (b >= 174 && b <= 255) continue;
        table.push_back({ (uint8_t)b, (uint32_t)(256 + n) });
        ++n;
    }
    return table;
}

inline std::unordered_map<std::string, int>
parse_vocab_json(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open vocab: " + path);
    std::stringstream ss; ss << f.rdbuf();
    const std::string s = ss.str();
    std::unordered_map<std::string, int> out;
    out.reserve(160000);
    size_t i = 0; const size_t n = s.size();
    auto skip_ws = [&]() {
        while (i < n && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
    };
    auto read_string = [&]() -> std::string {
        std::string r;
        if (i >= n || s[i] != '"') throw std::runtime_error("expected quote");
        ++i;
        while (i < n && s[i] != '"') {
            if (s[i] == '\\') {
                ++i;
                if (i >= n) break;
                const char e = s[i];
                switch (e) {
                case 'n': r += '\n'; break;
                case 't': r += '\t'; break;
                case 'r': r += '\r'; break;
                case 'b': r += '\b'; break;
                case 'f': r += '\f'; break;
                case '/': r += '/'; break;
                case '\\': r += '\\'; break;
                case '"': r += '"'; break;
                case 'u': {
                    if (i + 4 >= n) throw std::runtime_error("bad \\u");
                    uint32_t cp = 0;
                    for (int k = 1; k <= 4; ++k) {
                        const char h = s[i + k];
                        cp <<= 4;
                        if (h >= '0' && h <= '9')      cp |= (h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
                    }
                    i += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 < n &&
                        s[i + 1] == '\\' && s[i + 2] == 'u') {
                        uint32_t cp2 = 0;
                        for (int k = 3; k <= 6; ++k) {
                            const char h = s[i + k];
                            cp2 <<= 4;
                            if (h >= '0' && h <= '9')      cp2 |= (h - '0');
                            else if (h >= 'a' && h <= 'f') cp2 |= (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp2 |= (h - 'A' + 10);
                        }
                        if (cp2 >= 0xDC00 && cp2 <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (cp2 - 0xDC00);
                            i += 6;
                        }
                    }
                    r += utf8_encode(cp);
                    break;
                }
                default: r += e; break;
                }
                ++i;
            }
            else {
                r += s[i++];
            }
        }
        ++i;
        return r;
    };
    skip_ws();
    if (i >= n || s[i] != '{') throw std::runtime_error("expected {");
    ++i;
    while (true) {
        skip_ws();
        if (i >= n) break;
        if (s[i] == '}') { ++i; break; }
        const std::string key = read_string();
        skip_ws();
        if (i >= n || s[i] != ':') throw std::runtime_error("expected :");
        ++i; skip_ws();
        int v = 0; bool neg = false;
        if (s[i] == '-') { neg = true; ++i; }
        while (i < n && s[i] >= '0' && s[i] <= '9') {
            v = v * 10 + (s[i] - '0'); ++i;
        }
        if (neg) v = -v;
        out[key] = v;
        skip_ws();
        if (i < n && s[i] == ',') ++i;
    }
    return out;
}

inline std::unordered_map<std::string, int>
parse_merges_txt(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open merges: " + path);
    std::unordered_map<std::string, int> out;
    out.reserve(160000);
    std::string line;
    int rank = 0;
    bool first = true;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        if (first) {
            first = false;
            if (!line.empty() && line[0] == '#') continue;
        }
        if (line.empty()) continue;
        const size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        out[line.substr(0, sp) + " " + line.substr(sp + 1)] = rank++;
    }
    return out;
}

class Qwen2Tokenizer {
public:
    Qwen2Tokenizer(const std::string& vocab_path,
        const std::string& merges_path,
        const std::vector<std::pair<std::string, int>>& specials)
        : specials_(specials) {
        for (const auto& bu : byte_uni_table()) {
            byte_to_uni_[bu.first] = utf8_encode(bu.second);
            uni_to_byte_[bu.second] = bu.first;
        }
        encoder_ = parse_vocab_json(vocab_path);
        bpe_ranks_ = parse_merges_txt(merges_path);
        for (const auto& kv : encoder_) decoder_[kv.second] = kv.first;
        for (const auto& sp : specials_) {
            if (encoder_.find(sp.first) == encoder_.end())
                encoder_[sp.first] = sp.second;
            decoder_[sp.second] = sp.first;
            special_ids_.insert(sp.second);
        }
        std::fprintf(stderr, "[tokenizer] vocab=%zu merges=%zu specials=%zu\n",
            encoder_.size(), bpe_ranks_.size(), specials_.size());
    }

    int vocab_size() const { return static_cast<int>(encoder_.size()); }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;
        size_t last = 0;
        size_t pos = 0;
        while (pos < text.size()) {
            int matched_id = -1;
            size_t matched_len = 0;
            for (const auto& sp : specials_) {
                const size_t len = sp.first.size();
                if (text.compare(pos, len, sp.first) == 0 && len > matched_len) {
                    matched_id = sp.second;
                    matched_len = len;
                }
            }
            if (matched_id >= 0) {
                if (pos > last) append_segment(text, last, pos - last, ids);
                ids.push_back(matched_id);
                pos += matched_len;
                last = pos;
                continue;
            }
            ++pos;
        }
        if (last < text.size()) append_segment(text, last, text.size() - last, ids);
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string result, pending;
        for (int id : ids) {
            auto it = decoder_.find(id);
            if (it == decoder_.end()) continue;
            const std::string& tok = it->second;
            if (special_ids_.count(id)) {
                result += pending; pending.clear(); result += tok;
            }
            else {
                for (size_t i = 0; i < tok.size();) {
                    const auto cp_len = utf8_decode(tok, i);
                    auto bit = uni_to_byte_.find(cp_len.first);
                    if (bit != uni_to_byte_.end())
                        pending += static_cast<char>(bit->second);
                    else
                        pending += tok.substr(i, cp_len.second);
                    i += cp_len.second;
                }
            }
        }
        result += pending;
        return result;
    }

private:
    std::vector<std::pair<std::string, int>> specials_;
    std::unordered_set<int> special_ids_;
    std::unordered_map<std::string, int> encoder_;
    std::unordered_map<int, std::string> decoder_;
    std::unordered_map<std::string, int> bpe_ranks_;
    std::string byte_to_uni_[256];
    std::unordered_map<uint32_t, uint8_t> uni_to_byte_;

    void append_segment(const std::string& text, size_t off, size_t len,
        std::vector<int>& ids) const {
        const std::string seg = text.substr(off, len);
        const auto pieces = split_pieces(seg);
        for (const auto& p : pieces) {
            std::string bpe_input;
            bpe_input.reserve(p.second * 2);
            for (size_t k = 0; k < p.second; ++k)
                bpe_input += byte_to_uni_[static_cast<uint8_t>(seg[p.first + k])];
            for (const auto& w : bpe(bpe_input)) {
                auto it = encoder_.find(w);
                if (it != encoder_.end()) ids.push_back(it->second);
            }
        }
    }

    std::vector<std::pair<size_t, size_t>>
        split_pieces(const std::string& text) const {
        std::vector<std::pair<size_t, size_t>> pieces;
        size_t i = 0;
        const size_t n = text.size();
        while (i < n) {
            if (text[i] == '\'') {
                int matched = 0;
                if (i + 1 < n) {
                    const char c = (char)std::tolower((unsigned char)text[i + 1]);
                    if (c == 's' || c == 't' || c == 'm' || c == 'd') matched = 2;
                    else if (i + 2 < n) {
                        const char c2 = (char)std::tolower((unsigned char)text[i + 2]);
                        if ((c == 'r' && c2 == 'e') || (c == 'v' && c2 == 'e') || (c == 'l' && c2 == 'l'))
                            matched = 3;
                    }
                }
                if (matched > 0) { pieces.push_back({ i, (size_t)matched }); i += matched; continue; }
            }
            {
                const auto cl = utf8_decode(text, i);
                if (is_letter_cp(cl.first)) {
                    size_t j = i + cl.second;
                    while (j < n) {
                        const auto c2 = utf8_decode(text, j);
                        if (!is_letter_cp(c2.first)) break;
                        j += c2.second;
                    }
                    pieces.push_back({ i, j - i }); i = j; continue;
                }
                if (cl.first != '\r' && cl.first != '\n' && !is_digit_cp(cl.first)) {
                    const size_t j = i + cl.second;
                    if (j < n) {
                        const auto c2 = utf8_decode(text, j);
                        if (is_letter_cp(c2.first)) {
                            size_t k = j + c2.second;
                            while (k < n) {
                                const auto c3 = utf8_decode(text, k);
                                if (!is_letter_cp(c3.first)) break;
                                k += c3.second;
                            }
                            pieces.push_back({ i, k - i }); i = k; continue;
                        }
                    }
                }
            }
            {
                const auto cl = utf8_decode(text, i);
                if (is_digit_cp(cl.first)) { pieces.push_back({ i, (size_t)cl.second }); i += cl.second; continue; }
            }
            {
                size_t j = i; bool ok = false;
                if (text[i] == ' ') {
                    j = i + 1;
                    if (j < n) {
                        const auto c2 = utf8_decode(text, j);
                        if (!is_whitespace_cp(c2.first) && !is_letter_cp(c2.first) &&
                            !is_digit_cp(c2.first)) ok = true;
                    }
                }
                else {
                    const auto c0 = utf8_decode(text, i);
                    if (!is_whitespace_cp(c0.first) && !is_letter_cp(c0.first) &&
                        !is_digit_cp(c0.first)) {
                        ok = true; j = i;
                    }
                }
                if (ok) {
                    size_t k = j;
                    while (k < n) {
                        const auto c3 = utf8_decode(text, k);
                        if (is_whitespace_cp(c3.first) || is_letter_cp(c3.first) ||
                            is_digit_cp(c3.first)) break;
                        k += c3.second;
                    }
                    while (k < n && (text[k] == '\r' || text[k] == '\n')) ++k;
                    pieces.push_back({ i, k - i }); i = k; continue;
                }
            }
            {
                const auto cl = utf8_decode(text, i);
                if (is_whitespace_cp(cl.first)) {
                    std::vector<std::pair<size_t, size_t>> ws;
                    size_t k = i;
                    while (k < n) {
                        const auto c2 = utf8_decode(text, k);
                        if (!is_whitespace_cp(c2.first)) break;
                        ws.push_back({ k, (size_t)c2.second });
                        k += c2.second;
                    }
                    bool has_nl = false;
                    for (const auto& w : ws)
                        if (text[w.first] == '\n' || text[w.first] == '\r') { has_nl = true; break; }
                    if (has_nl) {
                        size_t last_nl_end = i;
                        for (const auto& w : ws)
                            if (text[w.first] == '\n' || text[w.first] == '\r')
                                last_nl_end = w.first + w.second;
                        pieces.push_back({ i, last_nl_end - i }); i = last_nl_end;
                    }
                    else {
                        const size_t cnt = ws.size();
                        size_t to_match = (k >= n) ? cnt : (cnt == 1 ? 1 : cnt - 1);
                        size_t end = i;
                        for (size_t c = 0; c < to_match; ++c) end += ws[c].second;
                        pieces.push_back({ i, end - i }); i = end;
                    }
                    continue;
                }
            }
            pieces.push_back({ i, 1 }); i += 1;
        }
        return pieces;
    }

    std::vector<std::string> bpe(const std::string& token) const {
        std::vector<std::string> word;
        for (size_t i = 0; i < token.size();) {
            const auto cp = utf8_decode(token, i);
            word.push_back(token.substr(i, cp.second));
            i += cp.second;
        }
        if (word.size() <= 1) return word;
        while (true) {
            int best_rank = INT32_MAX, best_idx = -1;
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                const std::string pair = word[i] + " " + word[i + 1];
                auto it = bpe_ranks_.find(pair);
                if (it != bpe_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second; best_idx = (int)i;
                }
            }
            if (best_idx == -1) break;
            word[best_idx] += word[best_idx + 1];
            word.erase(word.begin() + best_idx + 1);
            if (word.size() == 1) break;
        }
        return word;
    }
};