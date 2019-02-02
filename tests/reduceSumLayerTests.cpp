#include "abstractTensor.h"
#include "graphdl_ops.h"
#include "layerTests.h"
#include "reduceSum.h"

namespace
{
using namespace graphdl::core::layers;
using Param = std::tuple<UVec, int>;
using TestCase = std::tuple<Param, MemoryLocation>;

std::vector<Param> PARAMS = {
    // clang-format off
    {{1}, 1},
    {{1, 1}, 1},
    {{1, 1}, 2},
    {{1, 1, 1}, 1},
    {{1, 1, 1}, 2},
    {{1, 1, 1}, 3},
    {{2}, 1},
    {{2, 2}, 1},
    {{2, 2}, 2},
    {{2, 2, 2}, 1},
    {{2, 2, 2}, 2},
    {{2, 2, 2}, 3},
    {{2, 2, 2, 2}, 1},
    {{2, 2, 2, 2}, 2},
    {{2, 2, 2, 2}, 3},
    {{2, 2, 2, 2}, 4},
    {{2, 2, 2, 2, 2}, 1},
    {{2, 2, 2, 2, 2}, 2},
    {{2, 2, 2, 2, 2}, 3},
    {{2, 2, 2, 2, 2}, 4},
    {{2, 2, 2, 2, 2}, 5},
    {{2, 2, 2, 2, 2, 2}, 1},
    {{2, 2, 2, 2, 2, 2}, 2},
    {{2, 2, 2, 2, 2, 2}, 3},
    {{2, 2, 2, 2, 2, 2}, 4},
    {{2, 2, 2, 2, 2, 2}, 5},
    {{2, 2, 2, 2, 2, 2}, 6},
    {{10}, 1},
    {{10, 10}, 1},
    {{10, 10}, 2},
    {{10, 10, 10}, 1},
    {{10, 10, 10}, 2},
    {{10, 10, 10}, 3},
    {{2, 100}, 1},
    {{2, 100}, 2}
    // clang-format on
};

class ReduceSumTest : public LayerTest,
                      public testing::WithParamInterface<TestCase>
{
  public:
    void test(const TestCase& testCase)
    {
        UniformGen gen(seed);

        UVec shape = std::get<0>(std::get<0>(testCase));
        int numAxes = std::get<1>(std::get<0>(testCase));
        size_t outSize = 1, reduceSize = 1;
        for (unsigned i = 0; i < shape.size() - numAxes; ++i)
            outSize *= shape[i];
        for (unsigned i = shape.size() - numAxes; i < shape.size(); ++i)
            reduceSize *= shape[i];

        RefTensor input(shape, gen);
        RefTensor output(outputShape(testCase));

        for (size_t posY = 0; posY < outSize; ++posY)
        {
            output.at(posY) = 0.;
            for (size_t posX = 0; posX < reduceSize; ++posX)
                output.at(posY) += input.at(posY * reduceSize + posX);
        }

        LayerBuilder builder = [&](const HostVec& ins) {
            ITensorPtr in = createInput("in", shape, std::get<1>(testCase));
            ITensorPtr out = reduceSum(in, numAxes);
            initializeGraph();

            return HostVec({out->eval({{"in", ins[0]}})});
        };
        bool correct = runTest({input}, {output}, builder, 10e-5);
        EXPECT_TRUE(correct);
    }

    void testGradient(const TestCase& testCase)
    {
        UniformGen gen(seed);

        UVec shape = std::get<0>(std::get<0>(testCase));
        UVec outShape = outputShape(testCase);
        int numAxes = std::get<1>(std::get<0>(testCase));
        size_t outSize = 1, reduceSize = 1;
        for (unsigned i = 0; i < shape.size() - numAxes; ++i)
            outSize *= shape[i];
        for (unsigned i = shape.size() - numAxes; i < shape.size(); ++i)
            reduceSize *= shape[i];

        RefTensor input(shape, gen);
        RefTensor outputGrad(outShape, gen);
        RefTensor inputGrad(shape);

        for (size_t posY = 0; posY < outSize; ++posY)
            for (size_t posX = 0; posX < reduceSize; ++posX)
                inputGrad.at(posY * reduceSize + posX) = outputGrad.at(posY);

        LayerBuilder builder = [&](const HostVec& ins) {
            MemoryType type = memoryLocationToType(std::get<1>(testCase));
            Tensor::SPtr in = core::getDefaultGraph()->addInput(
                "in", createLayer<InputLayer>("in", shape, type));
            Tensor::SPtr outG = core::getDefaultGraph()->addInput(
                "outG", createLayer<InputLayer>("outG", outShape, type));
            Tensor::SPtr out = core::reduceSum(in, numAxes);
            Layer::SPtr layer =
                createLayer<ReduceSumGradientLayer>(in, numAxes, out, outG);
            Tensor::SPtr inG = layer->getOutputs()[0];
            initializeGraph();

            AbstractTensor::Ptr t = makeAbstractTensor(inG);
            return HostVec({t->eval({{"in", ins[0]}, {"outG", ins[1]}})});
        };
        bool correct = runTest({input, outputGrad}, {inputGrad}, builder);
        EXPECT_TRUE(correct);
    }

  private:
    UVec outputShape(const TestCase& testCase)
    {
        UVec shape = std::get<0>(std::get<0>(testCase));
        int numAxes = std::get<1>(std::get<0>(testCase));
        UVec outShape;
        for (unsigned i = 0; i < shape.size() - numAxes; ++i)
            outShape.push_back(shape[i]);

        return outShape;
    }
};

TEST_P(ReduceSumTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, ReduceSumTest,
                        Combine(ValuesIn(PARAMS), ValuesIn(LOCATIONS)));

class ReduceSumGradientTest : public ReduceSumTest
{
};
TEST_P(ReduceSumGradientTest, testAPI)
{
    testGradient(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, ReduceSumGradientTest,
                        Combine(ValuesIn(PARAMS), ValuesIn(LOCATIONS)));

}  // namespace
