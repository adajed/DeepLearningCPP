#include "activation.h"
#include "dll_ops.h"
#include "layerTests.h"

namespace
{
using namespace graphdl::core::layers;
using TestCase = std::tuple<Vec, Activation>;

std::vector<Vec> SHAPES = {
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
    {2, 2, 2, 2, 2, 2, 2},
    {2, 2, 2, 2, 2, 2, 2, 2},
    {2, 2, 2, 2, 2, 2, 2, 2, 2},
    {10, 10}
    // clang-format on
};

std::vector<Activation> OPS = {
    // clang-format off
    Activation::kRELU,
    Activation::kSIGMOID,
    Activation::kTANH,
    Activation::kSQUARE,
    Activation::kABS,
    Activation::kNEG,
    Activation::kRECIPROCAL
    // clang-format on
};

class ActivationTest : public LayerTest,
                       public testing::WithParamInterface<TestCase>
{
   public:
    void test(const TestCase& testCase)
    {
        setup(testCase);
        LayerBuilder builder = getBuilder(testCase);
        bool correct = runTest({mInput}, {mOutput}, builder);
        EXPECT_TRUE(correct);
    };

    void testGradient(const TestCase& testCase)
    {
        setupGradient(testCase);
        LayerBuilder builder = getGradientBuilder(testCase);
        bool correct =
            runTest({mInput, mOutputGrad}, {mGradient}, builder, 10e-4);
        EXPECT_TRUE(correct);
    };

   private:
    void setup(const TestCase& testCase)
    {
        UniformGen gen(0);
        mInput = RefTensor(std::get<0>(testCase));
        mOutput = RefTensor(std::get<0>(testCase));
        mInput.fillRandomly(gen);

        std::function<float(float)> fun;
        switch (std::get<1>(testCase))
        {
            case Activation::kRELU:
                fun = [](float x) {
                    if (x > 0.)
                        return x;
                    else
                        return 0.0f;
                };
                break;
            case Activation::kSIGMOID:
                fun = [](float x) { return 1. / (1. + std::exp(-x)); };
                break;
            case Activation::kTANH:
                fun = [](float x) { return std::tanh(x); };
                break;
            case Activation::kSQUARE:
                fun = [](float x) { return x * x; };
                break;
            case Activation::kABS:
                fun = [](float x) {
                    if (x > 0.)
                        return x;
                    else
                        return -x;
                };
                break;
            case Activation::kNEG:
                fun = [](float x) { return -x; };
                break;
            case Activation::kRECIPROCAL:
                fun = [](float x) { return 1. / x; };
                break;
        }

        for (std::size_t pos = 0; pos < mInput.count(); ++pos)
            mOutput.at(pos) = fun(mInput.at(pos));
    }

    void setupGradient(const TestCase& testCase)
    {
        UniformGen gen(0);
        mInput = RefTensor(std::get<0>(testCase));
        mOutputGrad = RefTensor(std::get<0>(testCase));
        mGradient = RefTensor(std::get<0>(testCase));
        mInput.fillRandomly(gen);
        mOutputGrad.fillRandomly(gen);

        std::function<float(float)> fun;
        switch (std::get<1>(testCase))
        {
            case Activation::kRELU:
                fun = [](float x) {
                    if (x > 0.)
                        return 1.;
                    else
                        return 0.;
                };
                break;
            case Activation::kSIGMOID:
                fun = [](float x) {
                    return std::exp(-x) /
                           ((1. + std::exp(-x)) * (1. + std::exp(-x)));
                };
                break;
            case Activation::kTANH:
                fun = [](float x) { return 1 - std::tanh(x) * std::tanh(x); };
                break;
            case Activation::kSQUARE:
                fun = [](float x) { return 2 * x; };
                break;
            case Activation::kABS:
                fun = [](float x) {
                    if (x > 0.)
                        return 1.;
                    else
                        return -1.;
                };
                break;
            case Activation::kNEG:
                fun = [](float x) { return -1; };
                break;
            case Activation::kRECIPROCAL:
                fun = [](float x) { return -1 / (x * x); };
                break;
        }

        for (std::size_t pos = 0; pos < mInput.count(); ++pos)
            mGradient.at(pos) = mOutputGrad.at(pos) * fun(mInput.at(pos));
    }

    LayerBuilder getBuilder(const TestCase& testCase)
    {
        return [&testCase](const HostVec& ins, const HostVec& outs) {
            ITensorPtr in = createInput("in", std::get<0>(testCase));
            ITensorPtr out;
            switch (std::get<1>(testCase))
            {
                case Activation::kRELU:
                    out = relu(in);
                    break;
                case Activation::kSIGMOID:
                    out = sigmoid(in);
                    break;
                case Activation::kTANH:
                    out = tanh(in);
                    break;
                case Activation::kSQUARE:
                    out = square(in);
                    break;
                case Activation::kABS:
                    out = abs(in);
                    break;
                case Activation::kNEG:
                    out = neg(in);
                    break;
                case Activation::kRECIPROCAL:
                    out = reciprocal(in);
                    break;
            }
            initializeGraph();

            out->eval({{"in", ins[0]}}, outs[0]);
        };
    }

    LayerBuilder getGradientBuilder(const TestCase& testCase)
    {
        return [&testCase](const HostVec& ins, const HostVec& outs) {
            Tensor::SPtr in =
                core::getDefaultGraph()->addInput("in", std::get<0>(testCase));
            Tensor::SPtr outG = core::getDefaultGraph()->addInput(
                "outG", std::get<0>(testCase));
            Tensor::SPtr out = createActivation(in, std::get<1>(testCase));
            Oper::SPtr oper = std::make_shared<ActivationGradientOper>(
                in, out, outG, std::get<1>(testCase));
            core::getDefaultGraph()->insertOperation(oper);
            ITensorPtr grad = ITensorPtr(oper->getOutputs()[0]);
            initializeGraph();
            grad->eval({{"in", ins[0]}, {"outG", ins[1]}}, outs[0]);
        };
    }

    RefTensor mInput, mOutput, mOutputGrad, mGradient;
};

TEST_P(ActivationTest, testAPI) { test(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, ActivationTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(OPS)));

class ActivationGradientTest : public ActivationTest
{
};
TEST_P(ActivationGradientTest, testAPI) { testGradient(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, ActivationGradientTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(OPS)));
}  // namespace
