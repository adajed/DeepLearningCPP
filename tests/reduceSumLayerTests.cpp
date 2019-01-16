#include "abstractTensor.h"
#include "graphdl_ops.h"
#include "layerTests.h"
#include "reduceSum.h"

namespace
{
using namespace graphdl::core::layers;
using TestCase = std::tuple<UVec, MemoryLocation>;

std::vector<UVec> SHAPES = {
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
        UniformGen gen(seed);

        RefTensor input(std::get<0>(testCase), gen);
        RefTensor output(TensorShape({}));

        output.at(0) = 0.;
        for (std::size_t pos = 0; pos < input.getCount(); ++pos)
            output.at(0) += input.at(pos);

        LayerBuilder builder = [&testCase](const HostVec& ins) {
            ITensorPtr in =
                createInput("in", std::get<0>(testCase), std::get<1>(testCase));
            ITensorPtr out = reduceSum(in);
            initializeGraph();

            return HostVec({out->eval({{"in", ins[0]}})});
        };
        bool correct = runTest({input}, {output}, builder, 10e-5);
        EXPECT_TRUE(correct);
    }

    void testGradient(const TestCase& testCase)
    {
        UniformGen gen(seed);
        RefTensor input(std::get<0>(testCase), gen);
        RefTensor outputGrad(TensorShape({}), gen);
        RefTensor inputGrad(std::get<0>(testCase));

        for (std::size_t pos = 0; pos < input.getCount(); ++pos)
            inputGrad.at(pos) = outputGrad.at(0);

        LayerBuilder builder = [&testCase](const HostVec& ins) {
            MemoryType type = memoryLocationToType(std::get<1>(testCase));
            Tensor::SPtr in = core::getDefaultGraph()->addInput(
                "in",
                createLayer<InputLayer>("in", std::get<0>(testCase), type));
            Tensor::SPtr outG = core::getDefaultGraph()->addInput(
                "outG", createLayer<InputLayer>("outG", Shape({}), type));
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

TEST_P(ReduceSumTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, ReduceSumTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(LOCATIONS)));

class ReduceSumGradientTest : public ReduceSumTest
{
};
TEST_P(ReduceSumGradientTest, testAPI)
{
    testGradient(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, ReduceSumGradientTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(LOCATIONS)));

}  // namespace
