#include "mini_test.h"
MINITEST_MAIN

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "quant.h"
#include "matmul.h"
#include "sampler.h"
#include "tokenizer.h"
#include "config.h"
#include "st_writer.h"
#include "st.h"
#include "tools.h"

namespace {

    std::vector<float> make_random_vec(size_t n, uint32_t seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        std::vector<float> v(n);
        for (auto& x : v) x = d(rng);
        return v;
    }

    class FileFixture : public ::testing::Test {
    protected:
        void SetUp() override {
            static std::atomic<int> counter{ 0 };
            dir = std::filesystem::temp_directory_path() /
                ("airun_t_" + std::to_string(counter.fetch_add(1)));
            std::filesystem::create_directories(dir);
        }
        void TearDown() override {
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
        }
        std::filesystem::path dir;
    };

} // namespace

TEST(QuantizePerChannel, ZeroMatrix_Handled) {
    std::vector<float> W(6, 0.0f);
    auto t = quantize_per_channel(W, 2, 3);
    EXPECT_EQ(t.rows, 2);
    EXPECT_EQ(t.cols, 3);
    ASSERT_EQ(t.scales.size(), 2u);
    for (float s : t.scales) EXPECT_GE(s, 1e-9f);      // clamped to floor
    for (int8_t v : t.data) EXPECT_EQ(static_cast<int>(v), 0);
}

TEST(QuantizePerChannel, ScaleIsMaxAbsOver127) {
    std::vector<float> W = { 1.f, -4.f, 3.f, -2.f, 0.5f, -0.5f, 0.25f, 0.125f };
    auto t = quantize_per_channel(W, 2, 4);
    EXPECT_NEAR(t.scales[0], 4.0f / 127.0f, 1e-7f);
    EXPECT_NEAR(t.scales[1], 0.5f / 127.0f, 1e-7f);
}

TEST(QuantizePerChannel, Empty_NoCrash) {
    std::vector<float> W;
    auto t = quantize_per_channel(W, 0, 0);
    EXPECT_EQ(t.rows, 0);
    EXPECT_EQ(t.cols, 0);
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.bytes(), 0u);
}

TEST(QuantizePerChannel, ClampsToSymmetricRange) {
    std::vector<float> W = { 127.f, -127.f, 1000.f, -1000.f };
    auto t = quantize_per_channel(W, 1, 4);
    for (int8_t v : t.data) {
        EXPECT_GE(static_cast<int>(v), -127);
        EXPECT_LE(static_cast<int>(v), 127);
    }
}

TEST(QuantizePerChannel, RoundTripApprox) {
    auto W = make_random_vec(64, 7);
    auto t = quantize_per_channel(W, 4, 16);
    for (int r = 0; r < 4; ++r) {
        const float s = t.scales[r];
        for (int c = 0; c < 16; ++c) {
            const float approx = static_cast<float>(t.data[r * 16 + c]) * s;
            EXPECT_NEAR(approx, W[r * 16 + c], s * 1.0001f + 1e-6f);
        }
    }
}

TEST(DotInt8Scalar, ZeroLength_ReturnsZero) {
    int8_t a = 1, b = 2;
    EXPECT_EQ(dot_int8_scalar(&a, &b, 0), 0);
}

TEST(DotInt8Scalar, SingleElement_Correct) {
    int8_t a = -5, b = 7;
    EXPECT_EQ(dot_int8_scalar(&a, &b, 1), -35);
}

TEST(DotInt8Scalar, VariousLengths_MatchReference) {
    for (int n : {7, 8, 15, 16, 31, 32, 33, 64, 100, 129}) {
        auto A = make_random_vec(n, 100 + n);
        auto B = make_random_vec(n, 200 + n);
        std::vector<int8_t> ai(n), bi(n);
        int64_t ref = 0;
        for (int i = 0; i < n; ++i) {
            ai[i] = static_cast<int8_t>(std::lround(A[i] * 100.0f));
            bi[i] = static_cast<int8_t>(std::lround(B[i] * 100.0f));
            ref += static_cast<int64_t>(ai[i]) * bi[i];
        }
        EXPECT_EQ(dot_int8_scalar(ai.data(), bi.data(), n),
            static_cast<int32_t>(ref)) << "n=" << n;
    }
}

TEST(DotInt8Scalar, Saturated_NoInt32Overflow) {
    const int n = 8192;
    std::vector<int8_t> a(n, 127), b(n, 127);
    const int64_t ref = static_cast<int64_t>(n) * 127 * 127;
    ASSERT_LT(ref, static_cast<int64_t>(INT32_MAX));
    EXPECT_EQ(dot_int8_scalar(a.data(), b.data(), n), static_cast<int32_t>(ref));
}

TEST(MatvecInt8, Identity_RoundTrip) {
    std::vector<float> I(16, 0.f);
    for (int i = 0; i < 4; ++i) I[i * 4 + i] = 1.f;
    auto W = quantize_per_channel(I, 4, 4);
    std::vector<float> x = { 1.f, -0.5f, 0.25f, 2.f };
    std::vector<float> out(4, 0.f);
    matvec_int8(x.data(), 4, W, nullptr, out.data(), 4);
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(out[i], x[i], 1e-2f);
}

TEST(MatvecInt8, AddsBias) {
    std::vector<float> I = { 1.f, 0.f, 0.f, 1.f };
    auto W = quantize_per_channel(I, 2, 2);
    std::vector<float> b = { 0.5f, -1.5f };
    std::vector<float> x = { 2.f, 3.f };
    std::vector<float> out(2, 0.f);
    matvec_int8(x.data(), 2, W, b.data(), out.data(), 2);
    EXPECT_NEAR(out[0], 2.5f, 1e-2f);
    EXPECT_NEAR(out[1], 1.5f, 1e-2f);
}

TEST(MatvecInt8Parallel, MatchesSequential) {
    // out_dim >= 2048 forces parallel path; nthreads defaults to hw.
    const int in_dim = 64, out_dim = 4096;
    auto Wf = make_random_vec(static_cast<size_t>(in_dim) * out_dim, 99);
    auto W = quantize_per_channel(Wf, out_dim, in_dim);
    auto x = make_random_vec(in_dim, 123);

    std::vector<float> seq(out_dim, 0.f), par(out_dim, 0.f);
    matvec_int8(x.data(), in_dim, W, nullptr, seq.data(), out_dim);
    matvec_int8_parallel(x.data(), in_dim, W, nullptr, par.data(), out_dim);
    for (int i = 0; i < out_dim; ++i) EXPECT_FLOAT_EQ(seq[i], par[i]);
}

TEST(RmsNorm, OnesEpsZero_Identity) {
    std::vector<float> x = { 1,1,1,1 }, g = { 1,1,1,1 }, out(4, 0.f);
    rms_norm(x.data(), g.data(), 4, 0.0f, out.data());
    for (float v : out) EXPECT_NEAR(v, 1.0f, 1e-5f);
}

TEST(RmsNorm, ZeroInput_ZeroOutput) {
    std::vector<float> x = { 0,0,0,0 }, g = { 1,1,1,1 }, out(4, 42.f);
    rms_norm(x.data(), g.data(), 4, 1e-6f, out.data());
    for (float v : out) EXPECT_FLOAT_EQ(v, 0.0f);
}

TEST(RmsNorm, GammaAppliesMultiplicatively) {
    std::vector<float> x = { 1,1,1,1 }, g = { 1,2,3,4 }, out(4, 0.f);
    rms_norm(x.data(), g.data(), 4, 0.0f, out.data());
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(out[i], static_cast<float>(i + 1), 1e-5f);
}

TEST(SiluInplace, Zero_Zero) {
    std::vector<float> x = { 0.0f };
    silu_inplace(x.data(), 1);
    EXPECT_FLOAT_EQ(x[0], 0.0f);
}

TEST(SiluInplace, One_KnownValue) {
    std::vector<float> x = { 1.0f };
    silu_inplace(x.data(), 1);
    EXPECT_NEAR(x[0], 1.0f / (1.0f + std::exp(-1.0f)), 1e-6f);
}

TEST(SoftmaxInplace, Uniform_Uniform) {
    std::vector<float> x = { 0,0,0,0 };
    softmax_inplace(x.data(), 4);
    for (float v : x) EXPECT_NEAR(v, 0.25f, 1e-6f);
}

TEST(SoftmaxInplace, SumsToOne) {
    std::vector<float> x = { 1.f, 2.f, 3.f, -1.f };
    softmax_inplace(x.data(), 4);
    float s = 0; for (float v : x) s += v;
    EXPECT_NEAR(s, 1.0f, 1e-6f);
}

TEST(SoftmaxInplace, LargeValues_NoOverflow) {
    std::vector<float> x = { 1000.f, 1001.f, 1002.f };
    softmax_inplace(x.data(), 3);
    float s = 0;
    for (float v : x) { EXPECT_TRUE(std::isfinite(v)); s += v; }
    EXPECT_NEAR(s, 1.0f, 1e-6f);
}

TEST(SoftmaxInplace, SingleElement_One) {
    std::vector<float> x = { 3.5f };
    softmax_inplace(x.data(), 1);
    EXPECT_FLOAT_EQ(x[0], 1.0f);
}

TEST(ApplyRope, PositionZero_Identity) {
    // No-op regardless of cached head_dim.
    init_rope_cache(64, 10000.0f);
    std::vector<float> v = { 1,2,3,4,5,6,7,8 };
    const std::vector<float> orig = v;
    apply_rope(v.data(), 0, 8, 0);
    for (size_t i = 0; i < v.size(); ++i) EXPECT_FLOAT_EQ(v[i], orig[i]);
}

TEST(ApplyRope, PreservesPairNorm) {
    // RoPE is an orthogonal rotation on each (i, i+half) pair.
    init_rope_cache(64, 10000.0f);
    std::vector<float> v = { 1.f, 2.f, 3.f, 4.f };
    const std::vector<float> orig = v;
    apply_rope(v.data(), 0, 4, 7);
    for (int i = 0; i < 2; ++i) {
        const float n0 = orig[i] * orig[i] + orig[i + 2] * orig[i + 2];
        const float n1 = v[i] * v[i] + v[i + 2] * v[i + 2];
        EXPECT_NEAR(n0, n1, 1e-5f);
    }
}

TEST(NumThreads, AtLeastOne) {
    EXPECT_GE(num_threads(), 1);
}

TEST(Sampler, TempZero_Argmax) {
    Sampler s; SamplerConfig c; c.temperature = 0.0f;
    std::vector<float> logits = { 1.f, 5.f, -3.f, 2.f };
    EXPECT_EQ(s.sample(logits.data(), 4, c, {}), 1);
}

TEST(Sampler, TopKOne_Argmax) {
    Sampler s; SamplerConfig c;
    c.temperature = 1.0f; c.top_k = 1; c.top_p = 0.9f;
    std::vector<float> logits = { 1.f, 5.f, -3.f, 2.f };
    for (int i = 0; i < 10; ++i)
        EXPECT_EQ(s.sample(logits.data(), 4, c, {}), 1);
}

TEST(Sampler, RepPenalty_PositiveDecreased) {
    Sampler s; SamplerConfig c;
    c.repetition_penalty = 2.0f;
    c.presence_penalty = 0; c.frequency_penalty = 0.0f;
    std::vector<float> logits = { 4.f, 4.f, 4.f };
    s.apply_penalties(logits.data(), 3, { 0 }, c);
    EXPECT_FLOAT_EQ(logits[0], 2.0f);
    EXPECT_FLOAT_EQ(logits[1], 4.0f);
    EXPECT_FLOAT_EQ(logits[2], 4.0f);
}

TEST(Sampler, RepPenalty_NegativeMoreNegative) {
    Sampler s; SamplerConfig c; c.repetition_penalty = 2.0f;
    std::vector<float> logits = { -4.f, 0.f };
    s.apply_penalties(logits.data(), 2, { 0 }, c);
    EXPECT_FLOAT_EQ(logits[0], -8.0f);
}

TEST(Sampler, EmptyRecent_NoChange) {
    Sampler s; SamplerConfig c; c.repetition_penalty = 2.0f;
    std::vector<float> logits = { 1,2,3 }, orig = logits;
    s.apply_penalties(logits.data(), 3, {}, c);
    for (size_t i = 0; i < logits.size(); ++i) EXPECT_FLOAT_EQ(logits[i], orig[i]);
}

TEST(Sampler, PresencePenalty_Subtracts) {
    Sampler s; SamplerConfig c;
    c.repetition_penalty = 1.0f; c.presence_penalty = 1;
    std::vector<float> logits = { 5.f, 5.f };
    s.apply_penalties(logits.data(), 2, { 0 }, c);
    EXPECT_FLOAT_EQ(logits[0], 4.0f);
    EXPECT_FLOAT_EQ(logits[1], 5.0f);
}

TEST(Sampler, FrequencyPenalty_MultipliesByCount) {
    Sampler s; SamplerConfig c;
    c.repetition_penalty = 1.0f; c.presence_penalty = 0;
    c.frequency_penalty = 1.0f;
    std::vector<float> logits = { 10.f, 5.f };
    s.apply_penalties(logits.data(), 2, { 0, 0, 0 }, c);   // 3 occurrences
    EXPECT_FLOAT_EQ(logits[0], 7.0f);
    EXPECT_FLOAT_EQ(logits[1], 5.0f);
}

TEST(Utf8, EncodeDecode_RoundTrip) {
    for (uint32_t cp : {0x41u, 0x7Au, 0xE9u, 0x2603u, 0x4E2Du, 0x1F600u}) {
        const std::string s = utf8_encode(cp);
        auto [dec, len] = utf8_decode(s, 0);
        EXPECT_EQ(dec, cp) << "cp=U+" << std::hex << cp;
        EXPECT_EQ(static_cast<size_t>(len), s.size());
    }
}

TEST(Utf8, Ascii_SingleByte) {
    const std::string s = utf8_encode(0x41);
    ASSERT_EQ(s.size(), 1u);
    EXPECT_EQ(static_cast<unsigned char>(s[0]), 0x41);
}

TEST(ByteUniTable, 256Entries) {
    EXPECT_EQ(byte_uni_table().size(), 256u);
}

TEST(ByteUniTable, UniqueCodepoints) {
    std::vector<uint32_t> cps;
    for (auto& [b, cp] : byte_uni_table()) cps.push_back(cp);
    std::sort(cps.begin(), cps.end());
    EXPECT_EQ(std::unique(cps.begin(), cps.end()), cps.end());
}

TEST_F(FileFixture, ParseVocabJson_Basic) {
    auto p = dir / "v.json";
    std::ofstream(p) << R"({"a": 0, "b": 1, "\u00e9": 2})";
    auto m = parse_vocab_json(p.string());
    EXPECT_EQ(m.at("a"), 0);
    EXPECT_EQ(m.at("b"), 1);
    EXPECT_EQ(m.at("\xC3\xA9"), 2);   // é
}

TEST_F(FileFixture, ParseVocabJson_Empty) {
    auto p = dir / "v.json";
    std::ofstream(p) << "{}";
    EXPECT_TRUE(parse_vocab_json(p.string()).empty());
}

TEST_F(FileFixture, ParseVocabJson_MissingFile_Throws) {
    EXPECT_THROW(parse_vocab_json((dir / "nope.json").string()), std::runtime_error);
}

TEST_F(FileFixture, ParseMerges_RanksIncreasing) {
    auto p = dir / "m.txt";
    std::ofstream(p) << "a b\nb c\nc d\n";
    auto m = parse_merges_txt(p.string());
    EXPECT_EQ(m.size(), 3u);
    EXPECT_LT(m.at("a b"), m.at("b c"));
    EXPECT_LT(m.at("b c"), m.at("c d"));
}

TEST(JsonParser, NumbersStrings_Basic) {
    std::string s = R"({"a": 42, "b": "hello", "c": 3.14, "d": true})";
    std::map<std::string, std::pair<std::string, std::string>> got;
    JsonParser p(s);
    p.parse_top_object([&](const std::string& k, const std::string& v,
        const std::string& t) { got[k] = { v, t }; });
    EXPECT_EQ(got["a"].first, "42");
    EXPECT_EQ(got["a"].second, "int");
    EXPECT_EQ(got["b"].first, "hello");
    EXPECT_EQ(got["b"].second, "string");
    EXPECT_EQ(got["c"].second, "float");
    EXPECT_EQ(got["d"].first, "true");
    EXPECT_EQ(got["d"].second, "bool");
}

TEST(JsonParser, SkipsNestedContainer) {
    std::string s = R"({"a": {"x": 1, "y": [2,3]}, "b": 7})";
    std::map<std::string, std::string> got;
    JsonParser p(s);
    p.parse_top_object([&](const std::string& k, const std::string& v,
        const std::string&) { got[k] = v; });
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got["b"], "7");
}

TEST(JsonParser, MissingBrace_Throws) {
    JsonParser p("not json");
    EXPECT_THROW(p.parse_top_object([](auto&, auto&, auto&) {}),
        std::runtime_error);
}

TEST(ToolsJson, ParsesStringArg) {
    std::string s = R"({"path": "a/b.txt"})";
    tools::json::Value v; size_t i = 0;
    ASSERT_TRUE(tools::json::parse_value(s, i, v, 0));
    ASSERT_EQ(v.t, tools::json::Value::Obj);
    const auto* p = tools::json::find(v, "path");
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(p->t, tools::json::Value::Str);
    EXPECT_EQ(p->s, "a/b.txt");
}

TEST(ToolsJson, ParsesNumberValue) {
    std::string s = R"({"n": 42})";
    tools::json::Value v; size_t i = 0;
    ASSERT_TRUE(tools::json::parse_value(s, i, v, 0));
    const auto* n = tools::json::find(v, "n");
    ASSERT_NE(n, nullptr);
    EXPECT_EQ(n->t, tools::json::Value::Num);
    EXPECT_EQ(n->s, "42");   // currently returns "{\"n\": 42"
}

TEST_F(FileFixture, Tools_SafeResolve_RejectsAbsolute) {
    tools::Config cfg; cfg.root = dir.string();
    EXPECT_THROW(tools::safe_resolve(cfg, "/etc/passwd"), std::runtime_error);
    EXPECT_THROW(tools::safe_resolve(cfg, dir.string() + "/x"), std::runtime_error);
}

TEST_F(FileFixture, Tools_SafeResolve_RejectsParentTraversal) {
    tools::Config cfg; cfg.root = dir.string();
    EXPECT_THROW(tools::safe_resolve(cfg, "../escape.txt"), std::runtime_error);
}

TEST_F(FileFixture, Tools_SafeResolve_AcceptsRelative) {
    tools::Config cfg; cfg.root = dir.string();
    EXPECT_NO_THROW(tools::safe_resolve(cfg, "a/b.txt"));
}

TEST_F(FileFixture, Tools_WriteRead_RoundTrip) {
    tools::Config cfg; cfg.root = dir.string();
    ASSERT_TRUE(tools::write_file(cfg, "hello.txt", "world").ok);
    auto r = tools::read_file(cfg, "hello.txt");
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.text, "world");
}

TEST_F(FileFixture, Tools_ReadMissing_Error) {
    tools::Config cfg; cfg.root = dir.string();
    auto r = tools::read_file(cfg, "nope.txt");
    EXPECT_FALSE(r.ok);
}

TEST_F(FileFixture, Tools_Append_AppendsToExisting) {
    tools::Config cfg; cfg.root = dir.string();
    tools::write_file(cfg, "log.txt", "a");
    tools::append_file(cfg, "log.txt", "b");
    EXPECT_EQ(tools::read_file(cfg, "log.txt").text, "ab");
}

TEST_F(FileFixture, Tools_DeleteDisabled_Fails) {
    tools::Config cfg; cfg.root = dir.string(); cfg.allow_delete = false;
    tools::write_file(cfg, "x.txt", "y");
    EXPECT_FALSE(tools::delete_file(cfg, "x.txt").ok);
}

TEST_F(FileFixture, Tools_WriteDisabled_Fails) {
    tools::Config cfg; cfg.root = dir.string(); cfg.allow_write = false;
    EXPECT_FALSE(tools::write_file(cfg, "x.txt", "y").ok);
}

TEST(ToolsFind, NoTag_False) {
    tools::Call c;
    EXPECT_FALSE(tools::find_tool_call("no tool here", c));
}

TEST(ToolsFind, BasicCall_Parsed) {
    std::string s =
        "text\n<tool_call>\n"
        R"({"name": "read_file", "arguments": {"path": "a.txt"}})"
        "\n</tool_call>";
    tools::Call c;
    ASSERT_TRUE(tools::find_tool_call(s, c));
    EXPECT_EQ(c.name, "read_file");
    EXPECT_NE(c.args_json.find("\"path\""), std::string::npos);
}

TEST(ToolsFind, MissingName_False) {
    std::string s = "<tool_call>{\"arguments\": {}}</tool_call>";
    tools::Call c;
    EXPECT_FALSE(tools::find_tool_call(s, c));
}

TEST_F(FileFixture, ToolsExecute_UnknownTool_Fails) {
    tools::Config cfg; cfg.root = dir.string();
    tools::Call c; c.name = "nope"; c.args_json = "{}";
    auto r = tools::execute(cfg, c);
    EXPECT_FALSE(r.ok);
}

TEST_F(FileFixture, SafeTensorsWriter_F32_RoundTrip) {
    auto p = (dir / "w.safetensors").string();
    SafeTensorsWriter w;
    std::vector<float> data = { 1, 2, 3, 4, 5, 6 };
    w.add_f32("w", { 2, 3 }, data.data(), 6);
    ASSERT_TRUE(w.save(p));

    SafeTensors st(p);
    ASSERT_TRUE(st.has("w"));
    const auto& info = st["w"];
    EXPECT_EQ(info.dtype, "F32");
    ASSERT_EQ(info.shape.size(), 2u);
    EXPECT_EQ(info.shape[0], 2);
    EXPECT_EQ(info.shape[1], 3);
    auto back = SafeTensors::to_f32(info);
    ASSERT_EQ(back.size(), 6u);
    for (size_t i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(back[i], data[i]);
}

TEST_F(FileFixture, SafeTensorsWriter_I8_RoundTrip) {
    auto p = (dir / "w8.safetensors").string();
    SafeTensorsWriter w;
    std::vector<int8_t> data = { 1, -2, 3, -4 };
    w.add_i8("w", { 2, 2 }, data.data(), 4);
    ASSERT_TRUE(w.save(p));

    SafeTensors st(p);
    const auto& info = st["w"];
    EXPECT_EQ(info.dtype, "I8");
    ASSERT_EQ(info.numel(), 4u);
    const int8_t* ptr = reinterpret_cast<const int8_t*>(info.ptr);
    for (size_t i = 0; i < 4; ++i) EXPECT_EQ(ptr[i], data[i]);
}

TEST(SafeTensorsWriter, ShapeSizeMismatch_Throws) {
    SafeTensorsWriter w;
    std::vector<float> d = { 1, 2, 3 };
    EXPECT_THROW(w.add_f32("x", { 2, 2 }, d.data(), 3), std::runtime_error);
}
