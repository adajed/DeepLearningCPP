#include "dll_errors.h"
#include "dll_ops.h"
#include "layerTests.h"
#include "matmul.h"

namespace
{
using namespace dll::core::layers;
using Vec = std::vector<unsigned>;
using TestCase = std::tuple<Vec, Vec>;

std::vector<TestCase> SHAPES = {
    // clang-format off
    {{1, 1}, {1, 1}},
    {{2, 2}, {2, 2}},
    {{2, 10}, {10, 3}},
    {{10, 10}, {10, 10}},
    {{1, 30}, {30, 1}},
    {{10, 100}, {100, 20}}, // big test
    // clang-format on
};

std::vector<TestCase> ERROR_SHAPES = {
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

        LayerBuilder builder = [&testCase](
                                   const std::vector<HostTensor>& ins,
                                   const std::vector<HostTensor>& outs) {
            dll::ITensorSPtr input1 =
                dll::createInput("i1", std::get<0>(testCase));
            dll::ITensorSPtr input2 =
                dll::createInput("i2", std::get<1>(testCase));
            dll::ITensorSPtr output = dll::matmul(input1, input2);
            dll::initializeGraph();
            output->eval({{"i1", ins[0]}, {"i2", ins[1]}}, outs[0]);
        };
        bool correct = runTest({mInput1, mInput2}, {mOutput}, builder);
        EXPECT_TRUE(correct);
    }

    void testGradient(const TestCase& testCase)
    {
        setupGradient(testCase);

        unsigned n = std::get<0>(testCase)[0];
        unsigned m = std::get<0>(testCase)[1];
        unsigned k = std::get<1>(testCase)[1];
        LayerBuilder builder = [n, m, k](const std::vector<HostTensor>& ins,
                                         const std::vector<HostTensor>& outs) {
            Tensor::SPtr i1 = core::getDefaultGraph()->addInput("i1", {n, m});
            Tensor::SPtr i2 = core::getDefaultGraph()->addInput("i2", {m, k});
            Tensor::SPtr outG =
                core::getDefaultGraph()->addInput("outG", {n, k});
            Tensor::SPtr out = matmul(i1, i2);
            Oper::SPtr oper =
                std::make_shared<MatmulGradientOper>(i1, i2, out, outG);
            core::getDefaultGraph()->insertOperation(oper);
            initializeGraph();

            std::vector<Tensor::SPtr> grads = oper->getOutputs();
            std::vector<ITensorSPtr> igrads = {ITensorSPtr(grads[0]),
                                               ITensorSPtr(grads[1])};

            eval(igrads, {{"i1", ins[0]}, {"i2", ins[1]}, {"outG", ins[2]}},
                 outs);
        };
        bool correct = runTest({mInput1, mInput2, mOutputGrad},
                               {mGradient1, mGradient2}, builder);
        EXPECT_TRUE(correct);
    }

    void testWrongShapes(const TestCase& testCase)
    {
        dll::ITensorSPtr input1 = dll::createInput("i1", std::get<0>(testCase));
        dll::ITensorSPtr input2 = dll::createInput("i2", std::get<1>(testCase));
        dll::ITensorSPtr output;
        EXPECT_THROW({ output = dll::matmul(input1, input2); },
                     dll::errors::NotMatchingShapesError);
    }

   private:
    RefTensor mInput1, mInput2, mOutput, mOutputGrad, mGradient1, mGradient2;

    void setup(const TestCase& testCase)
    {
        UniformGen gen(0);

        unsigned n = std::get<0>(testCase)[0];
        unsigned m = std::get<0>(testCase)[1];
        unsigned k = std::get<1>(testCase)[1];

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

        unsigned n = std::get<0>(testCase)[0];
        unsigned m = std::get<0>(testCase)[1];
        unsigned k = std::get<1>(testCase)[1];

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

TEST_P(MatmulTest, testAPI) { test(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, MatmulTest, ValuesIn(SHAPES));

class MatmulErrorsTest : public MatmulTest
{
};
TEST_P(MatmulErrorsTest, testWrongShapes) { testWrongShapes(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, MatmulErrorsTest, ValuesIn(ERROR_SHAPES));

class MatmulGradientTest : public MatmulTest
{
};
TEST_P(MatmulGradientTest, testAPI) { testGradient(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, MatmulGradientTest, ValuesIn(SHAPES));

}  // namespace
