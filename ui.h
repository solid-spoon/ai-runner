// cpp/ui.h — Minimal terminal UI helpers (Windows + POSIX).
//
// All on-screen decorations are plain ASCII so the UI renders identically
// on any terminal, code page, and font. No box-drawing, no braille,
// no fancy Unicode glyphs.
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
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <iostream>   // std::cin / std::getline

#ifdef _WIN32
#  include <windows.h>
#  include <io.h>
#  include <fcntl.h>
#else
#  include <sys/ioctl.h>
#  include <unistd.h>
#endif

namespace ui {

    // ??? ANSI colors (24-bit RGB) ????????????????????????????????????
    namespace color {
        inline const char* reset() { return "\x1b[0m"; }
        inline const char* bold() { return "\x1b[1m"; }
        inline const char* dim() { return "\x1b[2m"; }
        inline const char* italic() { return "\x1b[3m"; }
        inline const char* cyan() { return "\x1b[38;2;130;200;250m"; }
        inline const char* green() { return "\x1b[38;2;140;220;150m"; }
        inline const char* yellow() { return "\x1b[38;2;240;200;100m"; }
        inline const char* red() { return "\x1b[38;2;240;130;130m"; }
        inline const char* gray() { return "\x1b[38;2;120;120;130m"; }
        inline const char* gray2() { return "\x1b[38;2;80;80;90m"; }
        inline const char* magenta() { return "\x1b[38;2;200;140;240m"; }
        inline const char* white() { return "\x1b[38;2;230;230;235m"; }
    }

    // ??? Terminal setup ??????????????????????????????????????????????
    inline void init() {
#ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);

        // Put the C runtime streams into binary mode so that raw UTF-8 bytes
        // emitted by fputs/fwrite reach the terminal without CRLF translation
        // or code-page mangling.
        _setmode(_fileno(stdin), _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
        _setmode(_fileno(stderr), _O_BINARY);

        // Enable ANSI / VT escape sequences on the output console.
        HANDLE ho = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD mode = 0;
        if (GetConsoleMode(ho, &mode))
            SetConsoleMode(ho, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
        std::fputs("\x1b[?25h", stdout);
        std::fflush(stdout);
    }

    inline int term_width() {
#ifdef _WIN32
        CONSOLE_SCREEN_BUFFER_INFO info;
        if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info))
            return info.srWindow.Right - info.srWindow.Left + 1;
        return 100;
#else
        struct winsize w;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) return w.ws_col;
        return 100;
#endif
    }

    // ??? UTF-8 helpers ???????????????????????????????????????????????
    inline int char_width(uint32_t cp) {
        if (cp < 0x0300) return 1;
        if (cp >= 0x1100 && cp <= 0x115F) return 2;
        if (cp >= 0x2E80 && cp <= 0xA4CF) return 2;
        if (cp >= 0xAC00 && cp <= 0xD7A3) return 2;
        if (cp >= 0xF900 && cp <= 0xFAFF) return 2;
        if (cp >= 0xFE30 && cp <= 0xFE6F) return 2;
        if (cp >= 0xFF00 && cp <= 0xFF60) return 2;
        if (cp >= 0xFFE0 && cp <= 0xFFE6) return 2;
        if (cp >= 0x1F300 && cp <= 0x1F9FF) return 2;
        return 1;
    }

    inline int utf8_decode(const std::string& s, size_t i, uint32_t& cp) {
        const uint8_t c = static_cast<uint8_t>(s[i]);
        if (c < 0x80) { cp = c; return 1; }
        if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
            cp = (static_cast<uint32_t>(c & 0x1F) << 6) |
                (static_cast<uint8_t>(s[i + 1]) & 0x3F);
            return 2;
        }
        if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
            cp = (static_cast<uint32_t>(c & 0x0F) << 12) |
                ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 6) |
                (static_cast<uint8_t>(s[i + 2]) & 0x3F);
            return 3;
        }
        if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
            cp = (static_cast<uint32_t>(c & 0x07) << 18) |
                ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 12) |
                ((static_cast<uint8_t>(s[i + 2]) & 0x3F) << 6) |
                (static_cast<uint8_t>(s[i + 3]) & 0x3F);
            return 4;
        }
        cp = c;
        return 1;
    }

    // Number of terminal columns taken by an ANSI-decorated string.
    inline int str_width(const std::string& s) {
        int w = 0;
        size_t i = 0;
        while (i < s.size()) {
            if (s[i] == '\x1b') {
                // Skip an escape sequence.
                while (i < s.size() && s[i] != 'm') ++i;
                if (i < s.size()) ++i;
                continue;
            }
            uint32_t cp;
            const int l = utf8_decode(s, i, cp);
            w += char_width(cp);
            i += static_cast<size_t>(l);
        }
        return w;
    }

    // ??? User input ??????????????????????????????????????????????????
    //
    // Reads one line of user input as UTF-8.
    //
    // On Windows, `std::getline(std::cin, ...)` returns bytes in the console's
    // *code page*, not UTF-8 — so CJK and other non-ASCII characters get
    // destroyed. We avoid this by going through the wide-character API
    // `ReadConsoleW`, which always speaks UTF-16. We then convert to UTF-8.
    //
    // On POSIX, `std::getline` already returns UTF-8 bytes, so we use it directly.
    //
    // Multi-line pastes are supported: extra wide characters are buffered and
    // returned on subsequent calls.
    inline bool read_user_line(std::string& out) {
#ifdef _WIN32
        static std::wstring pending;

        HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
        const DWORD type = GetFileType(h);

        // Piped / redirected input (e.g. `echo ... | myprog`): bypass the wide
        // API and let the CRT hand us raw bytes.
        if (type != FILE_TYPE_CHAR) {
            return static_cast<bool>(std::getline(std::cin, out));
        }

        while (pending.find_first_of(L"\r\n") == std::wstring::npos) {
            wchar_t buf[1024];
            DWORD n = 0;
            if (!ReadConsoleW(h, buf,
                static_cast<DWORD>(sizeof(buf) / sizeof(buf[0])),
                &n, nullptr))
                return false;
            if (n == 0) {
                if (pending.empty()) return false;   // real EOF
                break;                               // EOF after partial data
            }
            pending.append(buf, n);
        }

        std::wstring line;
        const size_t cut = pending.find_first_of(L"\r\n");
        if (cut == std::wstring::npos) {
            line.swap(pending);
        }
        else {
            line = pending.substr(0, cut);
            size_t skip = cut + 1;
            if (pending[cut] == L'\r' && skip < pending.size() &&
                pending[skip] == L'\n')
                ++skip;
            pending.erase(0, skip);
        }

        if (line.empty()) { out.clear(); return true; }

        const int need = WideCharToMultiByte(
            CP_UTF8, 0, line.data(), static_cast<int>(line.size()),
            nullptr, 0, nullptr, nullptr);
        out.resize(static_cast<size_t>(need));
        WideCharToMultiByte(
            CP_UTF8, 0, line.data(), static_cast<int>(line.size()),
            out.data(), need, nullptr, nullptr);
        return true;
#else
        return static_cast<bool>(std::getline(std::cin, out));
#endif
    }

    // ??? Banner ??????????????????????????????????????????????????????
    inline void banner() {
        using namespace color;
        const int total = term_width();
        const int inner = std::min(total - 6, 76);

        auto hline = [&](char edge) {
            std::printf("  %s%c", gray(), edge);
            for (int i = 0; i < inner; ++i) std::printf("-");
            std::printf("%c%s\n", edge, reset());
        };

        auto row = [&](const std::string& content) {
            const int w = str_width(content);
            int pad = inner - 2 - w;
            if (pad < 0) pad = 0;
            std::printf("  %s|%s ", gray(), reset());
            std::fputs(content.c_str(), stdout);
            for (int i = 0; i < pad; ++i) std::printf(" ");
            std::printf(" %s|%s\n", gray(), reset());
        };

        std::printf("\n");
        hline('+');
        row("");
        row(std::string(bold()) + cyan() + "Qwen2.5-0.5B-Instruct" + reset());
        row(std::string(gray()) +
            "native C++ inference - SSE2 + threads" + reset());
        row("");
        row(std::string(gray()) + "type " + reset() + yellow() + "/help" +
            reset() + gray() + " for commands, " + reset() + yellow() + "/exit" +
            reset() + gray() + " to quit" + reset());
        row("");
        hline('+');
        std::printf("\n");
    }

    // ??? Message rendering ???????????????????????????????????????????
    inline void ai_msg_begin() {
        std::printf("  %s%saI %s %s>%s ", color::bold(), color::cyan(),
            color::reset(), color::gray(), color::reset());
        std::fflush(stdout);
    }

    inline void ai_msg_chunk(const std::string& text) {
        std::fputs(text.c_str(), stdout);
        std::fflush(stdout);
    }

    inline void ai_msg_end() { std::printf("\n"); }

    inline void info(const std::string& text) {
        std::printf("  %s%s%s\n", color::gray(), text.c_str(), color::reset());
    }

    inline void notice(const std::string& text) {
        std::printf("  %s%s%s\n", color::yellow(), text.c_str(), color::reset());
    }

    inline void error(const std::string& text) {
        std::printf("  %s[!] %s%s\n", color::red(), text.c_str(), color::reset());
    }

    // Prints the input prompt. The user's typed text is echoed by the
    // terminal itself, so no separate "echo the message" step is needed.
    inline void prompt() {
        std::printf("  %s%syou%s %s>%s ",
            color::bold(), color::green(), color::reset(),
            color::cyan(), color::reset());
        std::fflush(stdout);
    }

    inline void hint_line() {
        const int total = term_width();
        const int inner = std::min(total - 6, 76);

        std::printf("\n  %s-- commands ", color::gray2(), color::reset());
        for (int i = 0; i < std::max(0, inner - 15); ++i) std::printf("-");
        std::printf("%s\n", color::reset());

        std::printf("  %s|%s  ", color::gray2(), color::reset());
        std::printf("%s/reset%s  %s/clear%s  %s/temp%s <n>  %s/help%s  %s/exit%s",
            color::yellow(), color::reset(),
            color::yellow(), color::reset(),
            color::yellow(), color::reset(),
            color::yellow(), color::reset(),
            color::yellow(), color::reset());
        std::printf("\n  %s", color::gray2());
        for (int i = 0; i < inner; ++i) std::printf("-");
        std::printf("%s\n", color::reset());
    }

    // ??? Spinner (background thread) ?????????????????????????????????
    class Spinner {
    public:
        void start(const std::string& label) {
            stop_ = false;
            label_ = label;
            cur_ = 0;
            total_ = 0;
            th_ = std::thread([this] { loop(); });
        }

        void set_progress(int cur, int total) {
            cur_ = cur;
            total_ = total;
        }

        void stop() {
            stop_ = true;
            if (th_.joinable()) th_.join();
            std::printf("\r\x1b[2K");
            std::fflush(stdout);
        }

    private:
        void loop() {
            static const char kFrames[] = { '|', '/', '-', '\\' };
            int i = 0;
            while (!stop_) {
                const int c = cur_.load();
                const int t = total_.load();

                char prog[48] = { 0 };
                if (t > 0)
                    std::snprintf(prog, sizeof(prog), "  %d/%d", c, t);

                std::printf("\r\x1b[2K  %s%c%s  %s%s%s%s",
                    color::magenta(), kFrames[i & 3], color::reset(),
                    color::gray(), label_.c_str(), prog, color::reset());
                std::fflush(stdout);

                ++i;
                std::this_thread::sleep_for(std::chrono::milliseconds(80));
            }
        }

        std::atomic<bool> stop_{ false };
        std::thread th_;
        std::string label_;
        std::atomic<int> cur_{ 0 };
        std::atomic<int> total_{ 0 };
    };

} // namespace ui