#pragma once

#include <algorithm>   // std::max
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// ============================================================================
// Скелет «testing::Test» - совместимость с существующими фикстурами
// ============================================================================
namespace testing {
    class Test {
    public:
        virtual ~Test() = default;
    protected:
        virtual void SetUp() {}
        virtual void TearDown() {}
    };
}

namespace minitest {

    // ---------------------------------------------------------------------------
    // Учёт неудач
    // ---------------------------------------------------------------------------
    struct FatalFailure {};

    inline int& g_total_failures() { static int n = 0; return n; }
    inline int& g_current_failed() { static int n = 0; return n; }

    inline void report_failure(const char* file, int line, const std::string& msg) {
        std::cout << file << ":" << line << ": Failure\n" << msg << "\n";
        ++g_total_failures();
        ++g_current_failed();
    }

    // ---------------------------------------------------------------------------
    // SFINAE-хелпер: печатать значение только если для него есть operator<<.
    // Нужен, чтобы EXPECT_EQ на итераторах, std::filesystem::path и т.п.
    // компилировался (в gtest это делает UniversalPrinter).
    // ---------------------------------------------------------------------------
    template <typename T, typename = void>
    struct is_streamable : std::false_type {};

    template <typename T>
    struct is_streamable<T, std::void_t<
        decltype(std::declval<std::ostream&>() << std::declval<const T&>())>>
        : std::true_type {};

    template <typename T>
    void stream_or_placeholder(std::ostream& os, const T& v) {
        if constexpr (is_streamable<T>::value) os << v;
        else                                   os << "<unprintable>";
    }

    // ---------------------------------------------------------------------------
    // Объект-контекст утверждения. Живёт до конца полного выражения, поэтому
    // поддерживает цепочку `EXPECT_EQ(a,b) << "ctx=" << x;`
    // ---------------------------------------------------------------------------
    struct AssertionContext {
        const char* file;
        int line;
        bool fatal;
        bool passed;
        std::ostringstream os;

        AssertionContext(const char* f, int l, bool fat, bool ok)
            : file(f), line(l), fatal(fat), passed(ok) {}

        AssertionContext(const AssertionContext&) = delete;
        AssertionContext& operator=(const AssertionContext&) = delete;

        template <typename T>
        AssertionContext& operator<<(const T& v) {
            if (!passed) os << v;
            return *this;
        }
        // Для std::hex, std::endl и прочих манипуляторов
        AssertionContext& operator<<(std::ostream& (*m)(std::ostream&)) {
            if (!passed) m(os);
            return *this;
        }

        ~AssertionContext() noexcept(false) {
            if (!passed) {
                report_failure(file, line, os.str());
                // Бросаем FatalFailure только если нет активной раскрутки -
                // иначе получим std::terminate.
                if (fatal && std::uncaught_exceptions() == 0)
                    throw FatalFailure{};
            }
        }
    };

    // ---------------------------------------------------------------------------
    // Фабрики для бинарных сравнений (==, !=, <, <=, >, >=)
    // ---------------------------------------------------------------------------
    template <typename A, typename B>
    struct CmpAssertion : AssertionContext {
        CmpAssertion(const char* f, int l, bool fatal, bool ok,
            const char* as, const char* op, const char* bs, A a_, B b_)
            : AssertionContext(f, l, fatal, ok) {
            if (!ok) {
                os << "Expected: " << as << " " << op << " " << bs << "\n";
                os << "  LHS: "; stream_or_placeholder(os, a_);
                os << "\n  RHS: "; stream_or_placeholder(os, b_);
            }
        }
    };

    template <typename A, typename B, typename Cmp>
    CmpAssertion<std::decay_t<A>, std::decay_t<B>>
        make_cmp(const char* f, int l, bool fatal,
            const char* as, const char* op, const char* bs,
            A&& a, B&& b, Cmp&& cmp) {
        using TA = std::decay_t<A>;
        using TB = std::decay_t<B>;
        TA a_copy(std::forward<A>(a));
        TB b_copy(std::forward<B>(b));
        bool ok = cmp(a_copy, b_copy);
        return CmpAssertion<TA, TB>(f, l, fatal, ok, as, op, bs,
            std::move(a_copy), std::move(b_copy));
    }

    // ---------------------------------------------------------------------------
    // NEAR / FLOAT_EQ
    // ---------------------------------------------------------------------------
    struct NearAssertion : AssertionContext {
        NearAssertion(const char* f, int l, bool fatal, bool ok,
            const char* as, const char* bs,
            double a, double b, double tol)
            : AssertionContext(f, l, fatal, ok) {
            if (!ok) {
                os << "EXPECT_NEAR(" << as << ", " << bs << ", " << tol << ") failed\n"
                    << "  LHS: " << a << "\n"
                    << "  RHS: " << b << "\n"
                    << "  diff: " << std::fabs(a - b);
            }
        }
    };

    inline NearAssertion make_near(const char* f, int l, bool fatal,
        const char* as, const char* bs,
        double a, double b, double tol) {
        return NearAssertion(f, l, fatal, std::fabs(a - b) <= tol,
            as, bs, a, b, tol);
    }

    inline NearAssertion make_float_eq(const char* f, int l, bool fatal,
        const char* as, const char* bs,
        float a, float b) {
        const double tol = 1e-5 * (std::max(std::fabs((double)a),
            std::fabs((double)b)) + 1.0);
        return NearAssertion(f, l, fatal, std::fabs((double)a - (double)b) <= tol,
            as, bs, a, b, tol);
    }

    // ---------------------------------------------------------------------------
    // Реестр тестов
    // ---------------------------------------------------------------------------
    struct TestInfo {
        std::string suite;
        std::string name;
        std::function<void()> fn;
    };

    inline std::vector<TestInfo>& registry() {
        static std::vector<TestInfo> r;
        return r;
    }

    struct Registrar {
        Registrar(std::string s, std::string n, std::function<void()> f) {
            registry().push_back({ std::move(s), std::move(n), std::move(f) });
        }
    };

    inline int run_all(int /*argc*/ = 0, char** /*argv*/ = nullptr) {
        int passed = 0, failed = 0;
        for (auto& t : registry()) {
            g_current_failed() = 0;
            std::cout << "[ RUN      ] " << t.suite << "." << t.name << "\n";
            try {
                t.fn();
            }
            catch (const FatalFailure&) {
                // уже отчитались
            }
            catch (const std::exception& e) {
                report_failure("<unknown>", 0,
                    std::string("Uncaught exception: ") + e.what());
            }
            catch (...) {
                report_failure("<unknown>", 0, "Uncaught unknown exception");
            }
            if (g_current_failed()) {
                std::cout << "[  FAILED  ] " << t.suite << "." << t.name << "\n";
                ++failed;
            }
            else {
                std::cout << "[       OK ] " << t.suite << "." << t.name << "\n";
                ++passed;
            }
        }
        std::cout << "\n[==========] " << (passed + failed) << " tests ran.\n";
        std::cout << "[  PASSED  ] " << passed << " tests.\n";
        if (failed) std::cout << "[  FAILED  ] " << failed << " tests.\n";
        return failed ? 1 : 0;
    }

} // namespace minitest

// ============================================================================
// Макросы TEST / TEST_F
// ============================================================================
#define TEST(suite, name)                                                     \
    static void minitest_##suite##_##name##_body();                           \
    static ::minitest::Registrar minitest_reg_##suite##_##name(               \
        #suite, #name, &minitest_##suite##_##name##_body);                    \
    static void minitest_##suite##_##name##_body()

#define TEST_F(fixture, name)                                                 \
    struct minitest_fix_##fixture##_##name : public fixture {                 \
        void TestBody();                                                      \
        void RunBody() {                                                      \
            this->SetUp();                                                    \
            try { TestBody(); }                                               \
            catch (...) { try { this->TearDown(); } catch (...) {} throw; }   \
            this->TearDown();                                                 \
        }                                                                     \
    };                                                                        \
    static ::minitest::Registrar minitest_reg_##fixture##_##name(             \
        #fixture, #name, []() {                                               \
            minitest_fix_##fixture##_##name obj;                              \
            obj.RunBody();                                                    \
        });                                                                   \
    void minitest_fix_##fixture##_##name::TestBody()

// ============================================================================
// Утверждения
// ============================================================================
#define EXPECT_TRUE(cond)                                                     \
    (::minitest::AssertionContext(__FILE__, __LINE__, false,                  \
        static_cast<bool>(cond)) << "EXPECT_TRUE(" #cond ") failed")
#define EXPECT_FALSE(cond)                                                    \
    (::minitest::AssertionContext(__FILE__, __LINE__, false,                  \
        !static_cast<bool>(cond)) << "EXPECT_FALSE(" #cond ") failed")
#define ASSERT_TRUE(cond)                                                     \
    (::minitest::AssertionContext(__FILE__, __LINE__, true,                   \
        static_cast<bool>(cond)) << "ASSERT_TRUE(" #cond ") failed")

#define EXPECT_EQ(a, b)                                                       \
    ::minitest::make_cmp(__FILE__, __LINE__, false, #a, "==", #b, (a), (b),   \
        [](const auto& x, const auto& y) { return x == y; })
#define EXPECT_NE(a, b)                                                       \
    ::minitest::make_cmp(__FILE__, __LINE__, false, #a, "!=", #b, (a), (b),   \
        [](const auto& x, const auto& y) { return x != y; })
#define EXPECT_LT(a, b)                                                       \
    ::minitest::make_cmp(__FILE__, __LINE__, false, #a, "<",  #b, (a), (b),   \
        [](const auto& x, const auto& y) { return x <  y; })
#define EXPECT_LE(a, b)                                                       \
    ::minitest::make_cmp(__FILE__, __LINE__, false, #a, "<=", #b, (a), (b),   \
        [](const auto& x, const auto& y) { return x <= y; })
#define EXPECT_GT(a, b)                                                       \
    ::minitest::make_cmp(__FILE__, __LINE__, false, #a, ">",  #b, (a), (b),   \
        [](const auto& x, const auto& y) { return x >  y; })
#define EXPECT_GE(a, b)                                                       \
    ::minitest::make_cmp(__FILE__, __LINE__, false, #a, ">=", #b, (a), (b),   \
        [](const auto& x, const auto& y) { return x >= y; })

#define ASSERT_EQ(a, b)                                                       \
    ::minitest::make_cmp(__FILE__, __LINE__, true, #a, "==", #b, (a), (b),    \
        [](const auto& x, const auto& y) { return x == y; })
#define ASSERT_NE(a, b)                                                       \
    ::minitest::make_cmp(__FILE__, __LINE__, true, #a, "!=", #b, (a), (b),    \
        [](const auto& x, const auto& y) { return x != y; })
#define ASSERT_LT(a, b)                                                       \
    ::minitest::make_cmp(__FILE__, __LINE__, true, #a, "<", #b, (a), (b),     \
        [](const auto& x, const auto& y) { return x < y; })
#define ASSERT_LE(a, b)                                                       \
    ::minitest::make_cmp(__FILE__, __LINE__, true, #a, "<=", #b, (a), (b),    \
        [](const auto& x, const auto& y) { return x <= y; })
#define ASSERT_GT(a, b)                                                       \
    ::minitest::make_cmp(__FILE__, __LINE__, true, #a, ">", #b, (a), (b),     \
        [](const auto& x, const auto& y) { return x > y; })
#define ASSERT_GE(a, b)                                                       \
    ::minitest::make_cmp(__FILE__, __LINE__, true, #a, ">=", #b, (a), (b),    \
        [](const auto& x, const auto& y) { return x >= y; })

#define EXPECT_NEAR(a, b, tol)                                                \
    ::minitest::make_near(__FILE__, __LINE__, false, #a, #b,                  \
        static_cast<double>(a), static_cast<double>(b),                       \
        static_cast<double>(tol))
#define EXPECT_FLOAT_EQ(a, b)                                                 \
    ::minitest::make_float_eq(__FILE__, __LINE__, false, #a, #b,              \
        static_cast<float>(a), static_cast<float>(b))

// ---------------------------------------------------------------------------
// THROW / NO_THROW (без цепочки <<, как и в gtest)
// ---------------------------------------------------------------------------
#define EXPECT_THROW(stmt, exc_type)                                          \
    do {                                                                      \
        bool _mt_threw = false;                                               \
        try { stmt; }                                                         \
        catch (const exc_type&) { _mt_threw = true; }                         \
        catch (...) {}                                                        \
        if (!_mt_threw) {                                                     \
            ::minitest::report_failure(__FILE__, __LINE__,                    \
                "EXPECT_THROW(" #stmt ", " #exc_type ") failed: "             \
                "no " #exc_type " thrown");                                   \
        }                                                                     \
    } while (0)

#define EXPECT_NO_THROW(stmt)                                                 \
    do {                                                                      \
        try { stmt; }                                                         \
        catch (const std::exception& _e) {                                    \
            ::minitest::report_failure(__FILE__, __LINE__,                    \
                std::string("EXPECT_NO_THROW failed: ") + _e.what());         \
        } catch (...) {                                                       \
            ::minitest::report_failure(__FILE__, __LINE__,                    \
                "EXPECT_NO_THROW failed: unknown exception");                 \
        }                                                                     \
    } while (0)

// ============================================================================
// main() - линкуем в один .cpp
// ============================================================================
#define MINITEST_MAIN                                                         \
    int main(int argc, char** argv) {                                         \
        return ::minitest::run_all(argc, argv);                               \
    }
