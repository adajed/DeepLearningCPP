#include "dll_ops.h"
#include "layerTests.h"
#include "reduceSum.h"

namespace
{
using namespace dll::core::layers;
using TestCase = Vec;

std::vector<TestCase> SHAPES = {
    // clang-format off
    {},
    {1},
    {1, 1},
    {1, 1, 1},
    {2},
    {2, 2},
    {2, 2, 2},
    {2, 2, 2, 2},
    {2, 2, 2, 2, 2},
    {2, 2, 2, 2, 2, 2},
    {10},
    {10, 10},
    {10, 10, 10},
    {2, 100}
    // clang-format on
};

class ReduceSumTest : public LayerTest,
                      public testing::WithParamInterface<TestCase>
{
   public:
    void test(const TestCase& testCase)
    {
        UniformGen gen(0);

        RefTensor input(testCase);
        RefTensor output(TensorShape({}));
        input.fillRandomly(gen);

        output.at(0) = 0.;
        for (std::size_t pos = 0; pos < input.count(); ++pos)
            output.at(0) += input.at(pos);

        LayerBuilder builder = [&testCase](const HostVec& ins,
                                           const HostVec& outs) {
            ITensorSPtr in = createInput("in", testCase);
            ITensorSPtr out = reduceSum(in);
            initializeGraph();

            out->eval({{"in", ins[0]}}, outs[0]);
        };
        bool correct = runTest({input}, {output}, builder);
        EXPECT_TRUE(correct);
    }
};

TEST_P(ReduceSumTest, testAPI) { test(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, ReduceSumTest, ValuesIn(SHAPES));

}  // namespace
