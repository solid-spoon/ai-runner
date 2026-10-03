#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#undef min
#undef max

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <chrono>
#include <atomic>
#include <thread>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#endif

namespace ui {

    // ── Глифы ────────────────────────────────────────────────────
    inline constexpr const char* U_VBAR = "\xE2\x96\x8C"; // ▌
    inline constexpr const char* U_MIDDOT = "\xC2\xB7";     // ·
    inline constexpr const char* U_TL = "\xE2\x94\x8C"; // ┌
    inline constexpr const char* U_BL = "\xE2\x94\x94"; // └
    inline constexpr const char* U_V = "\xE2\x94\x82"; // │

    // ── Цвета. Инициализируются пустыми строками (безопасно до init()). ──
    inline const char* RESET = "";
    inline const char* BOLD = "";
    inline const char* DIM = "";
    inline const char* ITALIC = "";
    inline const char* ACCENT = "";
    inline const char* GREEN = "";
    inline const char* YELLOW = "";
    inline const char* RED = "";
    inline const char* TEAL = "";
    inline const char* MAUVE = "";
    inline const char* OVERLAY = "";
    inline const char* MUTED = "";
    inline const char* TEXT = "";

    inline bool& use_color() { static bool v = true; return v; }

    inline void init() {
#ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
        _setmode(_fileno(stdout), _O_BINARY);
        _setmode(_fileno(stderr), _O_BINARY);
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD mode = 0;
        if (hOut && hOut != INVALID_HANDLE_VALUE && GetConsoleMode(hOut, &mode))
            SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);

        use_color() = _isatty(_fileno(stdout)) != 0;
#else
        use_color() = isatty(fileno(stdout)) != 0;
#endif

        if (use_color()) {
            RESET = "\x1b[0m";
            BOLD = "\x1b[1m";
            DIM = "\x1b[2m";
            ITALIC = "\x1b[3m";
            ACCENT = "\x1b[38;2;137;180;250m";
            GREEN = "\x1b[38;2;166;227;161m";
            YELLOW = "\x1b[38;2;249;226;175m";
            RED = "\x1b[38;2;243;139;168m";
            TEAL = "\x1b[38;2;148;226;213m";
            MAUVE = "\x1b[38;2;203;166;247m";
            OVERLAY = "\x1b[38;2;108;112;134m";
            MUTED = "\x1b[38;2;88;91;112m";
            TEXT = "\x1b[38;2;205;214;244m";
        }
        // Иначе — остаются "" из объявления выше.
    }

    // ── banner / prompt ─────────────────────────────────────────
    inline void banner(const std::string& model_name) {
        std::fputc('\n', stdout);
        std::fprintf(stdout, "  %sairun%s  %s·%s  %s%s%s\n",
            BOLD, RESET, OVERLAY, RESET, ACCENT, model_name.c_str(), RESET);
        std::fprintf(stdout, "  %stype /help for commands, /exit to quit%s\n",
            DIM, RESET);
        std::fputc('\n', stdout);
        std::fflush(stdout);
    }

    inline void prompt() {
        std::fputc('\n', stdout);
        std::fprintf(stdout, "%s%s▌ you%s %s›%s ",
            GREEN, BOLD, RESET, OVERLAY, RESET);
        std::fflush(stdout);
    }

    // ── Стриминг сообщений с подсветкой код блоков ──────────────
        // ── Стриминг сообщений с подсветкой код блоков ──────────────
    namespace detail {
        struct StreamState {
            bool in_code = false;
            std::string line;
        };
        inline StreamState& state() { static StreamState s; return s; }

        inline bool is_fence_prefix(const std::string& line) {
            if (line.size() > 3) return line.compare(0, 3, "```") == 0;
            for (char c : line) if (c != '`') return false;
            return true;
        }

        // Длина (в байтах) начальной части `s`, которая состоит из
        // ПОЛНЫХ UTF-8 codepoint'ов. Если в хвосте `s` остался неполный
        // multi-byte символ, возвращает индекс, с которого начинается
        // этот неполный хвост — вызывающий должен придержать flush.
        inline size_t utf8_complete_len(const std::string& s) {
            size_t i = 0;
            while (i < s.size()) {
                const uint8_t b = static_cast<uint8_t>(s[i]);
                int need = 1;
                if (b < 0x80)               need = 1;
                else if ((b & 0xE0) == 0xC0) need = 2;
                else if ((b & 0xF0) == 0xE0) need = 3;
                else if ((b & 0xF8) == 0xF0) need = 4;
                else return i;   // невалидный lead byte — стоп
                if (i + static_cast<size_t>(need) > s.size()) return i;
                i += static_cast<size_t>(need);
            }
            return i;
        }

        inline void feed(const std::string& text) {
            auto& s = state();
            for (char c : text) {
                if (c == '\r') continue;

                if (c == '\n') {
                    if (s.line.size() >= 3 && s.line.compare(0, 3, "```") == 0) {
                        if (s.in_code) {
                            std::fprintf(stdout, "%s%s%s──%s\n",
                                MUTED, DIM, U_BL, RESET);
                            s.in_code = false;
                        }
                        else {
                            std::string lang = s.line.substr(3);
                            size_t a = lang.find_first_not_of(" \t");
                            size_t z = lang.find_last_not_of(" \t");
                            lang = (a == std::string::npos)
                                ? "" : lang.substr(a, z - a + 1);
                            std::fprintf(stdout, "%s%s%s── code%s%s%s%s\n",
                                MUTED, DIM, U_TL,
                                lang.empty() ? "" : " · ",
                                lang.c_str(), "", RESET);
                            s.in_code = true;
                        }
                    }
                    else if (s.in_code) {
                        std::fprintf(stdout, "%s%s%s %s%s%s\n",
                            MUTED, U_V, RESET, TEAL,
                            s.line.c_str(), RESET);
                    }
                    else {
                        if (!s.line.empty())
                            std::fprintf(stdout, "%s%s%s",
                                TEXT, s.line.c_str(), RESET);
                        std::fputc('\n', stdout);
                    }
                    s.line.clear();
                    continue;
                }

                s.line += c;

                if (utf8_complete_len(s.line) < s.line.size()) continue;
                if (s.in_code) continue;
                if (is_fence_prefix(s.line)) continue;

                std::fprintf(stdout, "%s%s%s", TEXT, s.line.c_str(), RESET);
                s.line.clear();
            }
            std::fflush(stdout);
        }
    } // namespace detail

    inline void ai_msg_begin() {
        detail::state().in_code = false;
        detail::state().line.clear();
        std::fprintf(stdout, "%s%s▌ ai%s\n", ACCENT, BOLD, RESET);
        std::fflush(stdout);
    }

    inline void ai_msg_chunk(const std::string& text) { detail::feed(text); }

    inline void ai_msg_end() {
        auto& s = detail::state();
        if (!s.line.empty()) {
            if (s.in_code) {
                std::fprintf(stdout, "%s%s%s %s%s%s\n",
                    MUTED, U_V, RESET, TEAL, s.line.c_str(), RESET);
            }
            else {
                std::fprintf(stdout, "%s%s%s\n", TEXT, s.line.c_str(), RESET);
            }
            s.line.clear();
        }
        if (s.in_code) {
            std::fprintf(stdout, "%s%s%s──%s\n", MUTED, DIM, U_BL, RESET);
            s.in_code = false;
        }
        std::fputc('\n', stdout);
        std::fflush(stdout);
    }

    // ── think ────────────────────────────────────────────────────
    inline void ai_msg_think_begin() {
        std::fprintf(stdout, "%s%s%s thinking%s\n", MUTED, ITALIC, U_MIDDOT, RESET);
        std::fflush(stdout);
    }

    inline void ai_msg_think_chunk(const std::string& text) {
        std::fprintf(stdout, "%s%s%s%s", MUTED, ITALIC, text.c_str(), RESET);
        std::fflush(stdout);
    }

    inline void ai_msg_think_end() {
        std::fputc('\n', stdout);
        std::fflush(stdout);
    }

    // ── Уведомления ──────────────────────────────────────────────
    inline void notice(const std::string& msg) {
        std::fprintf(stdout, "%s%s%s\n", YELLOW, msg.c_str(), RESET);
        std::fflush(stdout);
    }

    inline void error(const std::string& msg) {
        std::fprintf(stdout, "%s%s%s\n", RED, msg.c_str(), RESET);
        std::fflush(stdout);
    }

    inline void stats(const std::string& line) {
        std::fprintf(stdout, "%s%s%s %s%s\n",
            OVERLAY, DIM, U_MIDDOT, line.c_str(), RESET);
        std::fflush(stdout);
    }

    inline void help() {
        std::fprintf(stdout, "%s", OVERLAY);
        std::fprintf(stdout,
            "  /exit            quit\n"
            "  /reset           clear history + KV cache\n"
            "  /clear           clear screen and redraw banner\n"
            "  /think           toggle thinking mode\n"
            "  /temp <n>        set sampling temperature\n"
            "  /help            show this help\n");
        std::fprintf(stdout, "%s", RESET);
        std::fflush(stdout);
    }

    // ── Чтение строки (кросс платформенно) ──────────────────────
    inline bool read_user_line(std::string& out) {
        out.clear();
#ifdef _WIN32
        HANDLE hIn = GetStdHandle(STD_INPUT_HANDLE);
        DWORD ft = hIn ? GetFileType(hIn) : FILE_TYPE_UNKNOWN;
        if (ft == FILE_TYPE_CHAR) {
            static std::wstring pending;
            for (;;) {
                size_t nl = pending.find(L'\n');
                if (nl != std::wstring::npos) {
                    std::wstring w = pending.substr(0, nl);
                    pending.erase(0, nl + 1);
                    if (!w.empty() && w.back() == L'\r') w.pop_back();
                    int sz = WideCharToMultiByte(CP_UTF8, 0, w.data(),
                        static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
                    if (sz > 0) {
                        out.resize(static_cast<size_t>(sz));
                        WideCharToMultiByte(CP_UTF8, 0, w.data(),
                            static_cast<int>(w.size()), &out[0], sz, nullptr, nullptr);
                    }
                    return true;
                }
                wchar_t buf[256];
                DWORD got = 0;
                if (!ReadConsoleW(hIn, buf, 256, &got, nullptr) || got == 0)
                    return false;
                pending.append(buf, got);
            }
        }
#endif
        return static_cast<bool>(std::getline(std::cin, out));
    }

} // namespace ui