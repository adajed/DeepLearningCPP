#include "abstractTensor.h"
#include "graphdl_ops.h"
#include "layerTests.h"
#include "reduce.h"

namespace
{
using namespace graphdl::core::layers;
using Param = std::tuple<UVec, int>;
using TestCase = std::tuple<Param, ReduceType, MemoryLocation>;

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
    {{2, 100}, 2},
    {{100, 100}, 1},
    {{100, 100}, 2},
    {{100, 100, 10}, 1}, // big test for multiple reductions
    {{100, 100, 10}, 2} // big test for multiple reductions
    // clang-format on
};

std::vector<ReduceType> REDUCE_TYPES = {
    // clang-format off
    ReduceType::kSUM
    // clang-format on
};

std::ostream& operator<<(std::ostream& os, ReduceType t)
{
    switch (t)
    {
    case ReduceType::kSUM: return os << "SUM";
    }
}

class ReduceBackTest : public LayerTest,
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
            ITensorPtr in = createInput("in", shape, std::get<2>(testCase));
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
            MemoryType type = memoryLocationToType(std::get<2>(testCase));
            Tensor::SPtr in = core::getDefaultGraph()->addInput(
                "in", createLayer<InputLayer>("in", shape, type));
            Tensor::SPtr outG = core::getDefaultGraph()->addInput(
                "outG", createLayer<InputLayer>("outG", outShape, type));
            Tensor::SPtr out = core::reduceBack(in, numAxes, ReduceType::kSUM);
            Layer::SPtr layer = createLayer<ReduceBackGradientLayer>(
                in, out, outG, numAxes, ReduceType::kSUM);
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

class ReduceFrontTest : public LayerTest,
                        public testing::WithParamInterface<TestCase>
{
  public:
    void test(const TestCase& testCase)
    {
        UniformGen gen(seed);

        UVec shape = std::get<0>(std::get<0>(testCase));
        int numAxes = std::get<1>(std::get<0>(testCase));
        size_t outSize = 1, reduceSize = 1;
        for (unsigned i = 0; i < numAxes; ++i) reduceSize *= shape[i];
        for (unsigned i = numAxes; i < shape.size(); ++i) outSize *= shape[i];

        RefTensor input(shape, gen);
        RefTensor output(outputShape(testCase));

        for (size_t posY = 0; posY < outSize; ++posY)
        {
            output.at(posY) = 0.;
            for (size_t posX = 0; posX < reduceSize; ++posX)
                output.at(posY) += input.at(posX * outSize + posY);
        }

        LayerBuilder builder = [&](const HostVec& ins) {
            MemoryType type = memoryLocationToType(std::get<2>(testCase));
            Tensor::SPtr in = core::getDefaultGraph()->addInput(
                "in", createLayer<InputLayer>("in", shape, type));
            Tensor::SPtr out = core::reduceFront(in, numAxes, ReduceType::kSUM);
            initializeGraph();

            AbstractTensor::Ptr t = makeAbstractTensor(out);
            return HostVec({t->eval({{"in", ins[0]}})});
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
        for (unsigned i = 0; i < numAxes; ++i) reduceSize *= shape[i];
        for (unsigned i = numAxes; i < shape.size(); ++i) outSize *= shape[i];

        RefTensor input(shape, gen);
        RefTensor outputGrad(outShape, gen);
        RefTensor inputGrad(shape);

        for (size_t posY = 0; posY < outSize; ++posY)
            for (size_t posX = 0; posX < reduceSize; ++posX)
                inputGrad.at(posX * outSize + posY) = outputGrad.at(posY);

        LayerBuilder builder = [&](const HostVec& ins) {
            MemoryType type = memoryLocationToType(std::get<2>(testCase));
            Tensor::SPtr in = core::getDefaultGraph()->addInput(
                "in", createLayer<InputLayer>("in", shape, type));
            Tensor::SPtr outG = core::getDefaultGraph()->addInput(
                "outG", createLayer<InputLayer>("outG", outShape, type));
            Tensor::SPtr out = core::reduceFront(in, numAxes, ReduceType::kSUM);
            Layer::SPtr layer = createLayer<ReduceFrontGradientLayer>(
                in, out, outG, numAxes, ReduceType::kSUM);
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
        for (unsigned i = numAxes; i < shape.size(); ++i)
            outShape.push_back(shape[i]);

        return outShape;
    }
};

TEST_P(ReduceBackTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, ReduceBackTest,
                        Combine(ValuesIn(PARAMS), ValuesIn(REDUCE_TYPES),
                                ValuesIn(LOCATIONS)));

class ReduceBackGradientTest : public ReduceBackTest
{
};
TEST_P(ReduceBackGradientTest, testAPI)
{
    testGradient(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, ReduceBackGradientTest,
                        Combine(ValuesIn(PARAMS), ValuesIn(REDUCE_TYPES),
                                ValuesIn(LOCATIONS)));

TEST_P(ReduceFrontTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, ReduceFrontTest,
                        Combine(ValuesIn(PARAMS), ValuesIn(REDUCE_TYPES),
                                ValuesIn(LOCATIONS)));

class ReduceFrontGradientTest : public ReduceFrontTest
{
};
TEST_P(ReduceFrontGradientTest, testAPI)
{
    testGradient(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, ReduceFrontGradientTest,
                        Combine(ValuesIn(PARAMS), ValuesIn(REDUCE_TYPES),
                                ValuesIn(LOCATIONS)));

}  // namespace
