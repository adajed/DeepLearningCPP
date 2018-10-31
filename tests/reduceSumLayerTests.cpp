#include "dll_ops.h"
#include "layerTests.h"
#include "reduceSum.h"

namespace
{
using namespace graphdl::core::layers;
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
            ITensorPtr in = createInput("in", testCase);
            ITensorPtr out = reduceSum(in);
            initializeGraph();

            out->eval({{"in", ins[0]}}, outs[0]);
        };
        bool correct = runTest({input}, {output}, builder);
        EXPECT_TRUE(correct);
    }

    void testGradient(const TestCase& testCase)
    {
        UniformGen gen(0);
        RefTensor input(testCase);
        RefTensor outputGrad(TensorShape({}));
        RefTensor inputGrad(testCase);
        input.fillRandomly(gen);
        outputGrad.fillRandomly(gen);

        for (std::size_t pos = 0; pos < input.count(); ++pos)
            inputGrad.at(pos) = outputGrad.at(0);

        LayerBuilder builder = [&testCase](const HostVec& ins,
                                           const HostVec& outs) {
            Tensor::SPtr in = core::getDefaultGraph()->addInput("in", testCase);
            Tensor::SPtr outG = core::getDefaultGraph()->addInput("outG", {});
            Tensor::SPtr out = core::reduceSum(in);
            Oper::SPtr oper =
                std::make_shared<ReduceSumGradientOper>(in, out, outG);
            core::getDefaultGraph()->insertOperation(oper);
            Tensor::SPtr inG = oper->getOutputs()[0];
            initializeGraph();

            inG->eval({{"in", ins[0]}, {"outG", ins[1]}}, outs[0]);
        };
        bool correct = runTest({input, outputGrad}, {inputGrad}, builder);
        EXPECT_TRUE(correct);
    }
};

TEST_P(ReduceSumTest, testAPI) { test(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, ReduceSumTest, ValuesIn(SHAPES));

class ReduceSumGradientTest : public ReduceSumTest
{
};
TEST_P(ReduceSumGradientTest, testAPI) { testGradient(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, ReduceSumGradientTest, ValuesIn(SHAPES));

}  // namespace
