#include "abstractTensor.h"
#include "graphdl_ops.h"
#include "layerTests.h"
#include "matmul.h"

namespace
{
using namespace graphdl::core::layers;
using TestCase = std::tuple<std::pair<Vec, Vec>, MemoryLocation>;

std::vector<std::pair<Vec, Vec>> SHAPES = {
    // clang-format off
    {{1, 1}, {1, 1}},
    {{2, 2}, {2, 2}},
    {{2, 10}, {10, 3}},
    {{10, 10}, {10, 10}},
    {{1, 30}, {30, 1}},
    {{10, 100}, {100, 20}}, // big test
    // clang-format on
};

std::vector<std::pair<Vec, Vec>> ERROR_SHAPES = {
    // clang-format off
    {{1, 1}, {1}},
    {{2}, {2}},
    {{2, 2}, {2, 3, 3}},
    {{10, 1}, {2, 10}}
    // clang-format on
};

class MatmulTest : public LayerTest,
                   public testing::WithParamInterface<TestCase>
{
  public:
    void test(const TestCase& testCase)
    {
        setup(testCase);

        LayerBuilder builder = [&testCase](const HostVec& ins) {
            ITensorPtr input1 = createInput("i1", std::get<0>(testCase).first,
                                            std::get<1>(testCase));
            ITensorPtr input2 = createInput("i2", std::get<0>(testCase).second,
                                            std::get<1>(testCase));
            ITensorPtr output = matmul(input1, input2);
            initializeGraph();
            return HostVec({output->eval({{"i1", ins[0]}, {"i2", ins[1]}})});
        };
        bool correct = runTest({mInput1, mInput2}, {mOutput}, builder, 10e-5);
        EXPECT_TRUE(correct);
    }

    void testGradient(const TestCase& testCase)
    {
        setupGradient(testCase);

        unsigned n = std::get<0>(testCase).first[0];
        unsigned m = std::get<0>(testCase).first[1];
        unsigned k = std::get<0>(testCase).second[1];
        MemoryType type = memoryLocationToType(std::get<1>(testCase));
        LayerBuilder builder = [n, m, k, type](const HostVec& ins) {
            Tensor::SPtr i1 = core::getDefaultGraph()->addInput(
                "i1", createLayer<InputLayer>("i1", Shape({n, m}), type));
            Tensor::SPtr i2 = core::getDefaultGraph()->addInput(
                "i2", createLayer<InputLayer>("i2", Shape({m, k}), type));
            Tensor::SPtr outG = core::getDefaultGraph()->addInput(
                "outG", createLayer<InputLayer>("outG", Shape({n, k}), type));
            Tensor::SPtr out = matmul(i1, i2);
            Layer::SPtr layer =
                createLayer<MatmulGradientLayer>(i1, i2, out, outG);
            initializeGraph();

            std::vector<Tensor::SPtr> grads = layer->getOutputs();
            std::vector<ITensorPtr> igrads = {makeAbstractTensor(grads[0]),
                                              makeAbstractTensor(grads[1])};

            return eval(igrads,
                        {{"i1", ins[0]}, {"i2", ins[1]}, {"outG", ins[2]}});
        };
        bool correct = runTest({mInput1, mInput2, mOutputGrad},
                               {mGradient1, mGradient2}, builder, 10e-5);
        EXPECT_TRUE(correct);
    }

    void testWrongShapes(const TestCase& testCase)
    {
        ITensorPtr input1 = createInput("i1", std::get<0>(testCase).first,
                                        std::get<1>(testCase));
        ITensorPtr input2 = createInput("i2", std::get<0>(testCase).second,
                                        std::get<1>(testCase));
        ITensorPtr output;
        EXPECT_THROW({ output = matmul(input1, input2); }, std::runtime_error);
    }

  private:
    RefTensor mInput1, mInput2, mOutput, mOutputGrad, mGradient1, mGradient2;

    void setup(const TestCase& testCase)
    {
        UniformGen gen(0);

        unsigned n = std::get<0>(testCase).first[0];
        unsigned m = std::get<0>(testCase).first[1];
        unsigned k = std::get<0>(testCase).second[1];

        mInput1 = RefTensor({n, m});
        mInput2 = RefTensor({m, k});
        mOutput = RefTensor({n, k});

        mInput1.fillRandomly(gen);
        mInput2.fillRandomly(gen);

        // calculate reference output
        for (unsigned x = 0; x < n; ++x)
        {
            for (unsigned y = 0; y < k; ++y)
            {
                mOutput[{x, y}] = 0.;
                for (unsigned i = 0; i < m; ++i)
                    mOutput[{x, y}] += mInput1[{x, i}] * mInput2[{i, y}];
            }
        }
    }

    void setupGradient(const TestCase& testCase)
    {
        UniformGen gen(0);

        unsigned n = std::get<0>(testCase).first[0];
        unsigned m = std::get<0>(testCase).first[1];
        unsigned k = std::get<0>(testCase).second[1];

        mInput1 = RefTensor({n, m});
        mInput2 = RefTensor({m, k});
        mOutputGrad = RefTensor({n, k});
        mGradient1 = RefTensor({n, m});
        mGradient2 = RefTensor({m, k});

        mInput1.fillRandomly(gen);
        mInput2.fillRandomly(gen);
        mOutputGrad.fillRandomly(gen);

        // calculate reference gradient 1
        for (unsigned x = 0; x < n; ++x)
        {
            for (unsigned y = 0; y < m; ++y)
            {
                mGradient1[{x, y}] = 0.;
                for (unsigned i = 0; i < k; ++i)
                    mGradient1[{x, y}] += mInput2[{y, i}] * mOutputGrad[{x, i}];
            }
        }

        // calculate reference gradient 2
        for (unsigned x = 0; x < m; ++x)
        {
            for (unsigned y = 0; y < k; ++y)
            {
                mGradient2[{x, y}] = 0.;
                for (unsigned i = 0; i < n; ++i)
                    mGradient2[{x, y}] += mInput1[{i, x}] * mOutputGrad[{i, y}];
            }
        }
    }
};

TEST_P(MatmulTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, MatmulTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(LOCATIONS)));

class MatmulErrorsTest : public MatmulTest
{
};
TEST_P(MatmulErrorsTest, testWrongShapes)
{
    testWrongShapes(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, MatmulErrorsTest,
                        Combine(ValuesIn(ERROR_SHAPES), ValuesIn(LOCATIONS)));

class MatmulGradientTest : public MatmulTest
{
};
TEST_P(MatmulGradientTest, testAPI)
{
    testGradient(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, MatmulGradientTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(LOCATIONS)));

}  // namespace
