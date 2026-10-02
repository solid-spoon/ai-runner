// cpp/st_writer.h — minimal safetensors writer (F32 + I8 tensors).
#pragma once

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

class SafeTensorsWriter {
    struct Entry {
        std::string name;
        std::string dtype;              // "F32" or "I8"
        std::vector<int64_t> shape;
        std::vector<uint8_t> bytes;
    };
    std::vector<Entry> entries_;

public:
    void add_f32(const std::string& name,
        const std::vector<int64_t>& shape,
        const float* data, size_t n) {
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
        Entry e;
        e.name = name;
        e.dtype = "I8";
        e.shape = shape;
        e.bytes.assign(data, data + n);
        entries_.push_back(std::move(e));
    }

    bool save(const std::string& path) const {
        // ??? Build JSON header ???
        std::string json = "{";
        size_t offset = 0;
        for (size_t i = 0; i < entries_.size(); ++i) {
            const Entry& e = entries_[i];
            if (i > 0) json += ",";
            json += "\"" + e.name + "\":{";
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

        // Pad to 8-byte alignment (spec requirement).
        while (json.size() % 8 != 0) json += ' ';

        // ??? Write file ???
        FILE* f = nullptr;
        if (fopen_s(&f, path.c_str(), "wb") != 0 || !f) return false;

        const uint64_t header_len = json.size();
        std::fwrite(&header_len, 8, 1, f);
        std::fwrite(json.data(), 1, json.size(), f);
        for (const auto& e : entries_)
            std::fwrite(e.bytes.data(), 1, e.bytes.size(), f);
        std::fclose(f);
        return true;
    }
};