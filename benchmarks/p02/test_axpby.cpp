// P02 正确性测试：仅覆盖 CPU 路径，不包含任何 CUDA 依赖。
// GPU 路径的用例在独立的 p02_test_axpby_gpu 目标里，按 KERNELLAB_ENABLE_GPU_TESTS 注册。
#include "axpby.h"
#include "gpulab/testing/random.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

// 哨兵值：必须落在所有用例正确结果的取值范围之外。
// 本文件随机输入 x、y ∈ [-10, 10]，标量 |alpha|, |beta| <= 2，
// 因此 |alpha * x + beta * y| <= 40，永远不会等于 12345.0f。
constexpr float kSentinel = 12345.0f;

// 逐元素容限，按 RULES 的初始口径：abs_error <= 1e-5 + 1e-5 * |reference|。
constexpr float kAbsTolerance = 1e-5f;
constexpr float kRelTolerance = 1e-5f;

// 固定种子，写进报告即可复现。
constexpr std::mt19937::result_type kSeed = 20260803;

enum class InputKind { Fixed, Random };

const char* to_string(InputKind kind) {
    return kind == InputKind::Fixed ? "fixed" : "random";
}

struct TestCase {
    std::size_t n;
    float alpha;
    float beta;
    InputKind kind;
};

std::string describe(const TestCase& tc) {
    return "n=" + std::to_string(tc.n) + " alpha=" + std::to_string(tc.alpha)
        + " beta=" + std::to_string(tc.beta) + " input=" + to_string(tc.kind);
}

// 输入必须不对称：x 与 y 不同、标量之间不同，否则「alpha 与 beta 互换」这类 bug 会存活。
void fill_input(
    std::vector<float>& x,
    std::vector<float>& y,
    std::size_t n,
    InputKind kind,
    std::mt19937& rng) {
    if (kind == InputKind::Random) {
        gpulab::testing::fill_random_interleaved(
            {std::span<float>(x).first(n), std::span<float>(y).first(n)},
            rng);
        return;
    }
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i % 7) * 0.5f - 1.5f;
        y[i] = static_cast<float>(i % 5) * 0.25f - 0.75f;
    }
}

void report_failure(
    const TestCase& tc,
    std::size_t index,
    const char* reason,
    float expected,
    float actual) {
    const float tolerance = kAbsTolerance + kRelTolerance * std::abs(expected);
    // 失败详情走 std::cerr：它非缓冲，进程崩溃或被错误宏终止时诊断不会丢在 stdout 缓冲区里。
    std::cerr << "FAIL case{ " << describe(tc) << " } impl=axpby_host index=" << index
              << " reason=" << reason << " expected=" << expected << " actual=" << actual
              << " tolerance=" << tolerance << std::endl;
}

// 单条用例：哨兵预填 -> 调用 -> 先查覆盖、再查数值 -> 再查幂等（覆盖而非累加）。
// 返回 false 表示这条用例失败，但不中断其余用例。
bool run_case(const TestCase& tc, std::mt19937& rng) {
    std::vector<float> x(tc.n), y(tc.n), out(tc.n, kSentinel);
    fill_input(x, y, tc.n, tc.kind, rng);

    axpby_host(x.data(), y.data(), out.data(), tc.alpha, tc.beta, tc.n);

    bool ok = true;
    std::vector<float> first_pass = out;

    for (std::size_t i = 0; i < tc.n; ++i) {
        // 检查一：覆盖。仍等于哨兵 => 这个元素根本没被写过。
        if (out[i] == kSentinel) {
            report_failure(
                tc,
                i,
                "not_written(kSentinel survived)",
                tc.alpha * x[i] + tc.beta * y[i],
                out[i]);
            ok = false;
            continue;
        }
        // 检查二：数值。写了，但要在容限内等于参考值。
        const float reference = tc.alpha * x[i] + tc.beta * y[i];
        const float tolerance = kAbsTolerance + kRelTolerance * std::abs(reference);
        if (std::abs(out[i] - reference) > tolerance) {
            report_failure(tc, i, "value_mismatch", reference, out[i]);
            ok = false;
        }
    }

    // 检查三：out 必须被覆盖，不能累加旧值。同一输入再算一次，结果必须逐位相同。
    axpby_host(x.data(), y.data(), out.data(), tc.alpha, tc.beta, tc.n);
    for (std::size_t i = 0; i < tc.n && ok; ++i) {
        if (out[i] != first_pass[i]) {
            report_failure(tc, i, "not_idempotent(output accumulated)", first_pass[i], out[i]);
            ok = false;
        }
    }

    std::cout << (ok ? "PASS" : "FAIL") << " case{ " << describe(tc) << " }" << '\n';
    return ok;
}

// n=0 的合同：允许空指针，且不得写入任何内存。
// 传 nullptr 那一半靠 Sanitizer 证明「没解引用」，传真实缓冲那一半给出可判定的期望值。
bool run_zero_size_case() {
    bool ok = true;

    // 先做可判定的期望：同一合同下传真实缓冲（x、y、out 互不重叠），
    // 唯一合法的行为是「一个元素都不许改」。放在前面，这样违反合同会以可读失败呈现，
    // 而不是被后面的空指针崩溃掩盖。
    std::vector<float> x(4, 1.0f), y(4, -2.0f), guard(4, kSentinel);
    axpby_host(x.data(), y.data(), guard.data(), 1.5f, -0.5f, 0);
    for (std::size_t i = 0; i < guard.size(); ++i) {
        if (guard[i] != kSentinel) {
            std::cerr << "FAIL case{ n=0 } impl=axpby_host index=" << i
                      << " reason=unexpected_write_at_n_zero expected=" << kSentinel
                      << " actual=" << guard[i] << std::endl;
            ok = false;
        }
    }

    // 再验合同允许空指针这一半。若实现违反合同解引用，这里会崩溃，
    // 由 ASan / Compute Sanitizer 给出定位；放在最后是为了不掩盖上面那条可读失败。
    axpby_host(nullptr, nullptr, nullptr, 1.5f, -0.5f, 0);

    std::cout << (ok ? "PASS" : "FAIL") << " case{ n=0 contract=nullptr allowed, no writes }"
              << '\n';
    return ok;
}

} // namespace

bool test_axpby_cpu() {
    std::mt19937 rng(kSeed);
    std::size_t passed = 0;
    std::size_t failed = 0;

    // n=0 单独走合同用例，不混进矩阵（矩阵里 n=0 没有可比较的元素）。
    if (run_zero_size_case()) {
        ++passed;
    } else {
        ++failed;
    }

    // 矩阵：n 覆盖空/单元素/三元素手算长度/warp 边界/非 2 的幂长度；
    // 标量覆盖 0、1、负数与非平凡组合。共 6 * 4 * 2 = 48 条。
    const std::size_t sizes[] = {1, 3, 31, 32, 33, 1000};
    const float scalars[][2] = {{1.5f, -0.5f}, {1.0f, 1.0f}, {0.0f, 0.0f}, {-1.0f, 3.0f}};

    for (const std::size_t n : sizes) {
        for (const auto& sc : scalars) {
            for (const InputKind kind : {InputKind::Fixed, InputKind::Random}) {
                const TestCase tc{n, sc[0], sc[1], kind};
                if (run_case(tc, rng)) {
                    ++passed;
                } else {
                    ++failed;
                }
            }
        }
    }

    std::cout << "TOTAL=" << (passed + failed) << " PASSED=" << passed << " FAILED=" << failed
              << " seed=" << kSeed << '\n';
    return failed == 0;
}

int main() {
    // 逐条输出立即落盘：崩溃或 std::exit 时仍能保留已经跑过的用例记录。
    std::cout << std::unitbuf;
    return test_axpby_cpu() ? 0 : 1;
}
