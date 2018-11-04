#include "abstractTensor.h"
#include "graphdl_ops.h"
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
        for (std::size_t pos = 0; pos < input.getCount(); ++pos)
            output.at(0) += input.at(pos);

        LayerBuilder builder = [&testCase](const HostVec& ins) {
            ITensorPtr in = createInput("in", testCase, MemoryLocation::kHOST);
            ITensorPtr out = reduceSum(in);
            initializeGraph();

            return HostVec({out->eval({{"in", ins[0]}})});
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

        for (std::size_t pos = 0; pos < input.getCount(); ++pos)
            inputGrad.at(pos) = outputGrad.at(0);

        LayerBuilder builder = [&testCase](const HostVec& ins) {
            Tensor::SPtr in = core::getDefaultGraph()->addInput(
                "in", createLayer<InputLayer>("in", testCase, MemoryType::kHOST_MEMORY));
            Tensor::SPtr outG = core::getDefaultGraph()->addInput(
                "outG", createLayer<InputLayer>("outG", Shape({}), MemoryType::kHOST_MEMORY));
            Tensor::SPtr out = core::reduceSum(in);
            Layer::SPtr layer =
                createLayer<ReduceSumGradientLayer>(in, out, outG);
            Tensor::SPtr inG = layer->getOutputs()[0];
            initializeGraph();

            AbstractTensor::Ptr t = makeAbstractTensor(inG);
            return HostVec({t->eval({{"in", ins[0]}, {"outG", ins[1]}})});
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
