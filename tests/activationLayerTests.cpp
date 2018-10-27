#include "activation.h"
#include "dll_ops.h"
#include "layerTests.h"

namespace
{
using namespace dll::core::layers;
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

    LayerBuilder getBuilder(const TestCase& testCase)
    {
        return [&testCase](const std::vector<HostTensor>& ins,
                           const std::vector<HostTensor>& outs) {
            ITensorSPtr in = createInput("in", std::get<0>(testCase));
            ITensorSPtr out;
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

    RefTensor mInput, mOutput, mOutputGrad, mGradient;
};

TEST_P(ActivationTest, testAPI) { test(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, ActivationTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(OPS)));
}  // namespace
