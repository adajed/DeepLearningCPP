#include "assign.h"
#include "dll_errors.h"
#include "dll_ops.h"
#include "layerTests.h"

namespace
{
using TestCase = std::tuple<Vec>;
using ErrorTestCase = std::tuple<Vec, Vec>;

std::vector<TestCase> SHAPES = {
    // clang-format off
    {{}},
    {{1}},
    {{1, 1}},
    {{1, 1, 1}},
    {{1, 1, 1, 1}},
    {{1, 1, 1, 1, 1}},
    {{2}},
    {{2, 2}},
    {{2, 2, 2}},
    {{2, 2, 2, 2}},
    {{2, 2, 2, 2, 2}}
    // clang-format on
};

std::vector<ErrorTestCase> ERROR_SHAPES = {
    // clang-format off
    {{}, {1}},
    {{1}, {1, 1}},
    {{1, 1}, {1, 1, 1}},
    {{1, 1, 1}, {1, 1, 1, 1}},
    {{2}, {5}},
    {{2, 2}, {4}},
    {{5, 6}, {6, 5}},
    {{1, 10}, {10}}
    // clang-format on
};

class AssignTest : public LayerTest,
                   public testing::WithParamInterface<TestCase>
{
   public:
    void test(const TestCase& testCase)
    {
        UniformGen gen(0);
        RefTensor tensor(std::get<0>(testCase));
        tensor.fillRandomly(gen);

        LayerBuilder builder = [&testCase](HostVec ins, HostVec outs) {
            Tensor::SPtr in =
                core::getDefaultGraph()->addInput("in", std::get<0>(testCase));
            Tensor::SPtr w =
                core::getDefaultGraph()->addWeights("w", std::get<0>(testCase));
            Tensor::SPtr assign = core::assign(w, in);
            initializeGraph();

            HostTensor temp{nullptr, 0};
            assign->eval({{"in", ins[0]}}, temp);
            w->eval({}, outs[0]);
        };
        bool correct = runTest({tensor}, {tensor}, builder);
        EXPECT_TRUE(correct);
    }
};

class AssignErrorTest : public LayerTest,
                        public testing::WithParamInterface<ErrorTestCase>
{
   public:
    void test(const ErrorTestCase& testCase)
    {
        ITensorSPtr in = createInput("in", std::get<0>(testCase));
        ITensorSPtr w = createInput("w", std::get<1>(testCase));
        ITensorSPtr a;
        EXPECT_THROW({ a = assign(w, in); }, errors::NotMatchingShapesError);
    }
};

TEST_P(AssignTest, testAPI) { test(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, AssignTest, ValuesIn(SHAPES));

TEST_P(AssignErrorTest, test) { test(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerErrorTest, AssignErrorTest,
                        ValuesIn(ERROR_SHAPES));

}  // namespace
