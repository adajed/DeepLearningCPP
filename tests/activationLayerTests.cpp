#include "abstractTensor.h"
#include "activation.h"
#include "graphdl_ops.h"
#include "layerTests.h"

namespace
{
using namespace graphdl::core::layers;
using TestCase = std::tuple<UVec, Activation, MemoryLocation>;

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
    Activation::kRECIPROCAL,
    Activation::kLOG,
    Activation::kSQRT,
    Activation::kEXP
    // clang-format on
};

std::ostream& operator<<(std::ostream& os, Activation a)
{
    switch (a)
    {
    case Activation::kRELU: return os << "RELU";
    case Activation::kSIGMOID: return os << "SIGMOID";
    case Activation::kTANH: return os << "TANH";
    case Activation::kSQUARE: return os << "SQUARE";
    case Activation::kABS: return os << "ABS";
    case Activation::kNEG: return os << "NEG";
    case Activation::kRECIPROCAL: return os << "RECIPROCAL";
    case Activation::kLOG: return os << "LOG";
    case Activation::kSQRT: return os << "SQRT";
    case Activation::kEXP: return os << "EXP";
    }

    return os;
}

class ActivationTest : public LayerTest,
                       public testing::WithParamInterface<TestCase>
{
  public:
    void test(const TestCase& testCase)
    {
        setup(testCase);
        LayerBuilder builder = getBuilder(testCase);
        bool correct = runTest({mInput}, {mOutput}, builder, 10e-5);
        EXPECT_TRUE(correct);
    };

    void testGradient(const TestCase& testCase)
    {
        setupGradient(testCase);
        LayerBuilder builder = getGradientBuilder(testCase);
        bool correct =
            runTest({mInput, mOutputGrad}, {mGradient}, builder, 10e-3);
        EXPECT_TRUE(correct);
    };

  private:
    void setup(const TestCase& testCase)
    {
        UniformGen gen(seed);
        mInput = RefTensor(std::get<0>(testCase), gen);
        mOutput = RefTensor(std::get<0>(testCase));

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
        case Activation::kSQUARE: fun = [](float x) { return x * x; }; break;
        case Activation::kABS:
            fun = [](float x) {
                if (x > 0.)
                    return x;
                else
                    return -x;
            };
            break;
        case Activation::kNEG: fun = [](float x) { return -x; }; break;
        case Activation::kRECIPROCAL:
            fun = [](float x) { return 1. / x; };
            break;
        case Activation::kLOG:
            fun = [](float x) { return std::log(std::abs(x)); };
            break;
        case Activation::kSQRT:
            fun = [](float x) { return std::sqrt(std::abs(x)); };
            break;
        case Activation::kEXP: fun = [](float x) { return std::exp(x); }; break;
        }

        for (std::size_t pos = 0; pos < mInput.getCount(); ++pos)
            mOutput.at(pos) = fun(mInput.at(pos));
    }

    void setupGradient(const TestCase& testCase)
    {
        UniformGen gen(seed);
        mInput = RefTensor(std::get<0>(testCase), gen);
        mOutputGrad = RefTensor(std::get<0>(testCase), gen);
        mGradient = RefTensor(std::get<0>(testCase));

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
        case Activation::kSQUARE: fun = [](float x) { return 2 * x; }; break;
        case Activation::kABS:
            fun = [](float x) {
                if (x > 0.)
                    return 1.;
                else
                    return -1.;
            };
            break;
        case Activation::kNEG: fun = [](float x) { return -1; }; break;
        case Activation::kRECIPROCAL:
            fun = [](float x) { return -1 / (x * x); };
            break;
        case Activation::kLOG:
            fun = [](float x) { return 1 / std::abs(x); };
            break;
        case Activation::kSQRT:
            fun = [](float x) { return 1. / (2 * std::sqrt(std::abs(x))); };
            break;
        case Activation::kEXP: fun = [](float x) { return std::exp(x); }; break;
        }

        for (std::size_t pos = 0; pos < mInput.getCount(); ++pos)
            mGradient.at(pos) = mOutputGrad.at(pos) * fun(mInput.at(pos));
    }

    LayerBuilder getBuilder(const TestCase& testCase)
    {
        return [&testCase](const HostVec& ins) {
            ITensorPtr in =
                createInput("in", std::get<0>(testCase), std::get<2>(testCase));
            ITensorPtr out;
            switch (std::get<1>(testCase))
            {
            case Activation::kRELU: out = relu(in); break;
            case Activation::kSIGMOID: out = sigmoid(in); break;
            case Activation::kTANH: out = tanh(in); break;
            case Activation::kSQUARE: out = square(in); break;
            case Activation::kABS: out = abs(in); break;
            case Activation::kNEG: out = neg(in); break;
            case Activation::kRECIPROCAL: out = reciprocal(in); break;
            case Activation::kLOG: out = log(abs(in)); break;
            case Activation::kSQRT: out = sqrt(abs(in)); break;
            case Activation::kEXP: out = exp(in); break;
            }
            initializeGraph();

            return HostVec({out->eval({{"in", ins[0]}})});
        };
    }

    LayerBuilder getGradientBuilder(const TestCase& testCase)
    {
        return [&testCase](const HostVec& ins) {
            MemoryType type = memoryLocationToType(std::get<2>(testCase));
            Tensor::SPtr in = core::getDefaultGraph()->addInput(
                "in",
                createLayer<InputLayer>("in", std::get<0>(testCase), type));
            Tensor::SPtr outG = core::getDefaultGraph()->addInput(
                "outG",
                createLayer<InputLayer>("outG", std::get<0>(testCase), type));

            // make sure that input to log is positive
            if (std::get<1>(testCase) == Activation::kLOG ||
                std::get<1>(testCase) == Activation::kSQRT)
                in = abs(in);

            Tensor::SPtr out = createActivation(in, std::get<1>(testCase));
            Layer::SPtr layer = createLayer<ActivationGradientLayer>(
                in, out, outG, std::get<1>(testCase));
            ITensorPtr grad = makeAbstractTensor(layer->getOutputs()[0]);
            initializeGraph();
            return HostVec({grad->eval({{"in", ins[0]}, {"outG", ins[1]}})});
        };
    }

    RefTensor mInput, mOutput, mOutputGrad, mGradient;
};

TEST_P(ActivationTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, ActivationTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(OPS),
                                ValuesIn(LOCATIONS)));

class ActivationGradientTest : public ActivationTest
{
};
TEST_P(ActivationGradientTest, testAPI)
{
    testGradient(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, ActivationGradientTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(OPS),
                                ValuesIn(LOCATIONS)));
}  // namespace
