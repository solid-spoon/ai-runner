#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <stdexcept>

class SafeTensorsWriter {
    struct Entry {
        std::string name;
        std::string dtype;
        std::vector<int64_t> shape;
        std::vector<uint8_t> bytes;
    };
    std::vector<Entry> entries_;

    static std::string escape(const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            case '\r': out += "\\r";  break;
            default:   out += c;      break;
            }
        }
        return out;
    }

    static int64_t numel_of(const std::vector<int64_t>& shape) {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }

public:
    void add_f32(const std::string& name,
        const std::vector<int64_t>& shape,
        const float* data, size_t n) {
        // ѕроверка: shape и n должны совпадать, иначе запишем тензор
        // с неверными data_offsets и safetensors его отвергнет.
        if ((size_t)numel_of(shape) != n)
            throw std::runtime_error("add_f32: shape/numel mismatch for " + name);

        Entry e;
        e.name = name;
        e.dtype = "F32";
        e.shape = shape;
        e.bytes.resize(n * 4);
        std::memcpy(e.bytes.data(), data, n * 4);
        entries_.push_back(std::move(e));
    }

    void add_i8(const std::string& name,
        const std::vector<int64_t>& shape,
        const int8_t* data, size_t n) {
        if ((size_t)numel_of(shape) != n)
            throw std::runtime_error("add_i8: shape/numel mismatch for " + name);

        Entry e;
        e.name = name;
        e.dtype = "I8";
        e.shape = shape;
        e.bytes.assign(data, data + n);
        entries_.push_back(std::move(e));
    }

    bool save(const std::string& path) const {
        std::string json = "{";
        size_t offset = 0;
        for (size_t i = 0; i < entries_.size(); ++i) {
            const Entry& e = entries_[i];
            if (i > 0) json += ",";
            json += "\"" + escape(e.name) + "\":{";
            json += "\"dtype\":\"" + e.dtype + "\",";
            json += "\"shape\":[";
            for (size_t k = 0; k < e.shape.size(); ++k) {
                if (k > 0) json += ",";
                json += std::to_string(e.shape[k]);
            }
            json += "],\"data_offsets\":[";
            json += std::to_string(offset);
            json += ",";
            json += std::to_string(offset + e.bytes.size());
            json += "]}";
            offset += e.bytes.size();
        }
        json += "}";
        while (json.size() % 8 != 0) json += ' ';

        FILE* f = nullptr;
#ifdef _WIN32
        if (fopen_s(&f, path.c_str(), "wb") != 0 || !f) return false;
#else
        f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
#endif

        const uint64_t header_len = json.size();
        bool ok = std::fwrite(&header_len, 8, 1, f) == 1;
        ok = ok && std::fwrite(json.data(), 1, json.size(), f) == json.size();
        for (const auto& e : entries_) {
            if (!ok) break;
            ok = std::fwrite(e.bytes.data(), 1, e.bytes.size(), f) == e.bytes.size();
        }
        std::fclose(f);
        return ok;
    }
};