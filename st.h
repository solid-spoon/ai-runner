// cpp/st.h — safetensors reader with mmap (Windows + POSIX)
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// ─── TensorInfo ─────────────────────────────────────────────
struct TensorInfo {
    std::string name;
    std::string dtype;                 // "F32", "BF16", "F16", "I32", "I64"
    std::vector<int64_t> shape;
    size_t offset_begin = 0;
    size_t offset_end = 0;
    const uint8_t* ptr = nullptr;

    size_t numel() const {
        size_t n = 1;
        for (auto s : shape) n *= (size_t)s;
        return n;
    }
    size_t elem_size() const {
        if (dtype == "F32" || dtype == "I32") return 4;
        if (dtype == "F16" || dtype == "BF16") return 2;
        if (dtype == "I8" || dtype == "U8")   return 1;
        if (dtype == "I64") return 8;
        throw std::runtime_error("unknown dtype: " + dtype);
    }
    size_t nbytes() const { return numel() * elem_size(); }
};

// ─── SafeTensors ────────────────────────────────────────────
class SafeTensors {
    void* mmap_ptr_ = nullptr;
    size_t mmap_size_ = 0;
#ifdef _WIN32
    HANDLE hFile_ = nullptr;
    HANDLE hMap_ = nullptr;
#else
    int fd_ = -1;
#endif
    const uint8_t* base_ = nullptr;
    std::unordered_map<std::string, TensorInfo> tensors_;

public:
    explicit SafeTensors(const std::string& path) {
#ifdef _WIN32
        hFile_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
            NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile_ == INVALID_HANDLE_VALUE)
            throw std::runtime_error("cannot open: " + path);

        LARGE_INTEGER sz;
        GetFileSizeEx(hFile_, &sz);
        mmap_size_ = (size_t)sz.QuadPart;

        hMap_ = CreateFileMappingA(hFile_, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap_) throw std::runtime_error("CreateFileMapping failed");

        mmap_ptr_ = MapViewOfFile(hMap_, FILE_MAP_READ, 0, 0, 0);
        if (!mmap_ptr_) throw std::runtime_error("MapViewOfFile failed");
#else
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open: " + path);
        struct stat st;
        if (fstat(fd_, &st) != 0) throw std::runtime_error("fstat failed");
        mmap_size_ = (size_t)st.st_size;
        mmap_ptr_ = mmap(NULL, mmap_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mmap_ptr_ == MAP_FAILED) throw std::runtime_error("mmap failed");
#endif
        base_ = (const uint8_t*)mmap_ptr_;
        parse();
    }

    ~SafeTensors() {
#ifdef _WIN32
        if (mmap_ptr_) UnmapViewOfFile(mmap_ptr_);
        if (hMap_)     CloseHandle(hMap_);
        if (hFile_)    CloseHandle(hFile_);
#else
        if (mmap_ptr_) munmap(mmap_ptr_, mmap_size_);
        if (fd_ >= 0)  close(fd_);
#endif
    }

    SafeTensors(const SafeTensors&) = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;

    const TensorInfo& operator[](const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) throw std::runtime_error("tensor not found: " + name);
        return it->second;
    }
    bool has(const std::string& name) const { return tensors_.count(name) > 0; }
    const auto& all() const { return tensors_; }
    size_t size() const { return tensors_.size(); }

    // ─── Конвертация в Float32 ───
    static std::vector<float> to_f32(const TensorInfo& t) {
        size_t n = t.numel();
        std::vector<float> out(n);
        if (t.dtype == "F32") {
            std::memcpy(out.data(), t.ptr, n * 4);
        }
        else if (t.dtype == "BF16") {
            const uint16_t* p = (const uint16_t*)t.ptr;
            for (size_t i = 0; i < n; i++) {
                uint32_t u = (uint32_t)p[i] << 16;
                std::memcpy(&out[i], &u, 4);
            }
        }
        else if (t.dtype == "F16") {
            const uint16_t* p = (const uint16_t*)t.ptr;
            for (size_t i = 0; i < n; i++) {
                uint16_t h = p[i];
                uint32_t s = (h >> 15) & 1;
                uint32_t e = (h >> 10) & 0x1f;
                uint32_t f = h & 0x3ff;
                float v;
                if (e == 0) {
                    v = (f / 1024.0f) * (float)std::pow(2.0, -14);
                }
                else if (e == 31) {
                    uint32_t u = (s << 31) | 0x7f800000u | (f << 13);
                    std::memcpy(&v, &u, 4);
                }
                else {
                    uint32_t u = (s << 31) | ((e + 112) << 23) | (f << 13);
                    std::memcpy(&v, &u, 4);
                }
                out[i] = v;
            }
        }
        else {
            throw std::runtime_error("cannot convert dtype: " + t.dtype);
        }
        return out;
    }

private:
    void parse() {
        uint64_t header_len;
        std::memcpy(&header_len, base_, 8);
        std::string h((const char*)base_ + 8, (size_t)header_len);
        const uint8_t* data_base = base_ + 8 + header_len;

        size_t i = 0;
        auto skip_ws = [&]() {
            while (i < h.size() &&
                (h[i] == ' ' || h[i] == '\n' || h[i] == '\r' || h[i] == '\t')) i++;
        };
        auto expect = [&](char c) {
            if (i >= h.size() || h[i] != c)
                throw std::runtime_error(std::string("expected '") + c +
                    "' at " + std::to_string(i));
            i++;
        };
        auto read_string = [&]() {
            expect('"');
            std::string s;
            while (i < h.size() && h[i] != '"') s += h[i++];
            expect('"');
            return s;
        };
        // Проматывает любое JSON-значение целиком (для неизвестных полей и __metadata__)
        auto skip_value = [&]() {
            skip_ws();
            if (i >= h.size()) return;
            char c = h[i];
            if (c == '"') {
                i++;
                while (i < h.size() && h[i] != '"') {
                    if (h[i] == '\\' && i + 1 < h.size()) i++;
                    i++;
                }
                if (i < h.size()) i++;
                return;
            }
            if (c == '{' || c == '[') {
                int depth = 0; bool in_str = false;
                while (i < h.size()) {
                    char ch = h[i];
                    if (in_str) {
                        if (ch == '\\') { i += 2; continue; }
                        if (ch == '"') in_str = false;
                    }
                    else {
                        if (ch == '"') in_str = true;
                        else if (ch == '{' || ch == '[') depth++;
                        else if (ch == '}' || ch == ']') {
                            depth--;
                            if (depth == 0) { i++; return; }
                        }
                    }
                    i++;
                }
                return;
            }
            // число / true / false / null
            while (i < h.size() && h[i] != ',' && h[i] != '}' && h[i] != ']') i++;
        };

        skip_ws(); expect('{');
        while (true) {
            skip_ws();
            if (i >= h.size()) break;
            if (h[i] == '}') { i++; break; }

            std::string key = read_string();
            skip_ws(); expect(':'); skip_ws();

            // Пропускаем служебное поле safetensors
            if (key == "__metadata__") {
                skip_value();
                skip_ws(); if (i < h.size() && h[i] == ',') i++;
                continue;
            }

            TensorInfo t;
            t.name = key;
            expect('{');
            while (true) {
                skip_ws();
                if (h[i] == '}') { i++; break; }
                std::string field = read_string();
                skip_ws(); expect(':'); skip_ws();

                if (field == "dtype") {
                    t.dtype = read_string();
                }
                else if (field == "shape") {
                    expect('[');
                    while (true) {
                        skip_ws();
                        if (h[i] == ']') { i++; break; }
                        int64_t v = 0;
                        while (i < h.size() && h[i] >= '0' && h[i] <= '9')
                            v = v * 10 + (h[i++] - '0');
                        t.shape.push_back(v);
                        skip_ws(); if (h[i] == ',') i++;
                    }
                }
                else if (field == "data_offsets") {
                    expect('[');
                    size_t vals[2] = { 0, 0 }; int vi = 0;
                    while (true) {
                        skip_ws();
                        if (h[i] == ']') { i++; break; }
                        size_t v = 0;
                        while (i < h.size() && h[i] >= '0' && h[i] <= '9')
                            v = v * 10 + (h[i++] - '0');
                        if (vi < 2) vals[vi++] = v;
                        skip_ws(); if (h[i] == ',') i++;
                    }
                    t.offset_begin = vals[0];
                    t.offset_end = vals[1];
                }
                else {
                    skip_value();
                }
                skip_ws(); if (i < h.size() && h[i] == ',') i++;
            }
            t.ptr = data_base + t.offset_begin;
            tensors_[key] = std::move(t);

            skip_ws(); if (i < h.size() && h[i] == ',') i++;
        }
    }
};