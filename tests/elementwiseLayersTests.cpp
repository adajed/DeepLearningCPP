#include "abstractTensor.h"
#include "elementwise.h"
#include "graphdl_ops.h"
#include "layerTests.h"

namespace
{
using namespace graphdl::core::layers;

using TestCase = std::tuple<Vec, Elementwise>;
using ErrorTestCase = std::tuple<std::tuple<Vec, Vec>, Elementwise>;

Shape shape(const TestCase& testCase) { return std::get<0>(testCase); }
Elementwise op(const TestCase& testCase) { return std::get<1>(testCase); }

std::vector<Vec> SHAPES = {
    // clang-format off
    {},
    {1},
    {1, 1},
    {2},
    {2, 2},
    {2, 2, 2},
    {2, 2, 2, 2},
    {2, 2, 2, 2, 2},
    {2, 2, 2, 2, 2, 2},
    {20, 20}
    // clang-format on
};

std::vector<std::tuple<Vec, Vec>> ERROR_SHAPES = {
    // clang-format off
    {{}, {2}},
    {{2}, {5}},
    {{2, 3}, {2, 5}},
    {{3, 4}, {4, 3}},
    {{2, 2}, {2}},
    {{2}, {2, 2}},
    {{2, 2}, {2, 2, 2}},
    {{2, 2, 2}, {2, 2, 2, 2}}
    // clang-format on
};

std::vector<Elementwise> OPS = {
    // clang-format off
    Elementwise::kADD,
    Elementwise::kSUB,
    Elementwise::kMUL,
    Elementwise::kDIV
    // clang-format on
};

class ElementwiseTest : public LayerTest,
                        public testing::WithParamInterface<TestCase>
{
   public:
    void test(const TestCase& testCase)
    {
        setup(testCase);

        LayerBuilder builder = getBuilder(testCase);
        bool correct = runTest({mInput1, mInput2}, {mOutput}, builder);

        EXPECT_TRUE(correct);
    }

    void testGradient(const TestCase& testCase)
    {
        setupGradient(testCase);
        LayerBuilder builder = getGradientBuilder(testCase);
        bool correct = runTest({mInput1, mInput2, mOutputGrad},
                               {mGradient1, mGradient2}, builder);

        EXPECT_TRUE(correct);
    }

   protected:
    RefTensor mInput1, mInput2, mOutput, mOutputGrad, mGradient1, mGradient2;

   private:
    void setup(const TestCase& testCase)
    {
        UniformGen gen(0);

        // allocate memory
        mInput1 = RefTensor(shape(testCase));
        mInput2 = RefTensor(shape(testCase));
        mOutput = RefTensor(shape(testCase));

        // fill inputs with random data
        mInput1.fillRandomly(gen);
        mInput2.fillRandomly(gen);

        std::function<float(float, float)> f;
        switch (op(testCase))
        {
            case Elementwise::kADD:
                f = [](float f1, float f2) { return f1 + f2; };
                break;
            case Elementwise::kSUB:
                f = [](float f1, float f2) { return f1 - f2; };
                break;
            case Elementwise::kMUL:
                f = [](float f1, float f2) { return f1 * f2; };
                break;
            case Elementwise::kDIV:
                f = [](float f1, float f2) { return f1 / f2; };
                break;
        }
        // calculate reference output
        for (std::size_t i = 0; i < mInput1.getCount(); ++i)
            mOutput.at(i) = f(mInput1.at(i), mInput2.at(i));
    }

    void setupGradient(const TestCase& testCase)
    {
        UniformGen gen(0);

        mInput1 = RefTensor(shape(testCase));
        mInput2 = RefTensor(shape(testCase));
        mOutputGrad = RefTensor(shape(testCase));
        mGradient1 = RefTensor(shape(testCase));
        mGradient2 = RefTensor(shape(testCase));

        mInput1.fillRandomly(gen);
        mInput2.fillRandomly(gen);
        mOutputGrad.fillRandomly(gen);

        std::function<float(float, float)> fun1;
        std::function<float(float, float)> fun2;
        switch (op(testCase))
        {
            case Elementwise::kADD:
                fun1 = [](float f1, float f2) { return 1.; };
                fun2 = [](float f1, float f2) { return 1.; };
                break;
            case Elementwise::kSUB:
                fun1 = [](float f1, float f2) { return 1.; };
                fun2 = [](float f1, float f2) { return -1.; };
                break;
            case Elementwise::kMUL:
                fun1 = [](float f1, float f2) { return f2; };
                fun2 = [](float f1, float f2) { return f1; };
                break;
            case Elementwise::kDIV:
                fun1 = [](float f1, float f2) { return 1. / f2; };
                fun2 = [](float f1, float f2) { return -f1 / (f2 * f2); };
                break;
        }

        for (std::size_t i = 0; i < mInput1.getCount(); ++i)
        {
            mGradient1.at(i) =
                mOutputGrad.at(i) * fun1(mInput1.at(i), mInput2.at(i));
            mGradient2.at(i) =
                mOutputGrad.at(i) * fun2(mInput1.at(i), mInput2.at(i));
        }
    }

    LayerBuilder getBuilder(const TestCase& testCase)
    {
        return [testCase](const HostVec& ins) {
            ITensorPtr input1 =
                createInput("input1", shape(testCase), MemoryLocation::kHOST);
            ITensorPtr input2 =
                createInput("input2", shape(testCase), MemoryLocation::kHOST);
            ITensorPtr output;
            switch (op(testCase))
            {
                case Elementwise::kADD:
                    output = input1 + input2;
                    break;
                case Elementwise::kSUB:
                    output = input1 - input2;
                    break;
                case Elementwise::kMUL:
                    output = input1 * input2;
                    break;
                case Elementwise::kDIV:
                    output = input1 / input2;
                    break;
            }
            initializeGraph();
            return HostVec(
                {output->eval({{"input1", ins[0]}, {"input2", ins[1]}})});
        };
    }

    LayerBuilder getGradientBuilder(const TestCase& testCase)
    {
        return [&testCase](const HostVec& ins) {
            Tensor::SPtr in1 = core::getDefaultGraph()->addInput(
                "in1", createLayer<InputLayer>("in1", shape(testCase),
                                               MemoryType::kHOST_MEMORY));
            Tensor::SPtr in2 = core::getDefaultGraph()->addInput(
                "in2", createLayer<InputLayer>("in2", shape(testCase),
                                               MemoryType::kHOST_MEMORY));
            Tensor::SPtr outG = core::getDefaultGraph()->addInput(
                "outG", createLayer<InputLayer>("outG", shape(testCase),
                                                MemoryType::kHOST_MEMORY));
            Tensor::SPtr output = createElementwise(in1, in2, op(testCase));
            Layer::SPtr layer = createLayer<ElementwiseGradientLayer>(
                in1, in2, output, outG, op(testCase));
            initializeGraph();
            std::vector<Tensor::SPtr> grads = layer->getOutputs();

            std::vector<ITensorPtr> igrads = {makeAbstractTensor(grads[0]),
                                              makeAbstractTensor(grads[1])};
            return eval(igrads,
                        {{"in1", ins[0]}, {"in2", ins[1]}, {"outG", ins[2]}});
        };
    }
};

class ElementwiseGradientTest : public ElementwiseTest
{
};

class ElementwiseErrorTest : public LayerTest,
                             public testing::WithParamInterface<ErrorTestCase>
{
   public:
    void test(const ErrorTestCase& testCase)
    {
        std::tuple<Vec, Vec> shapes = std::get<0>(testCase);
        ITensorPtr input1 =
            createInput("input1", std::get<0>(shapes), MemoryLocation::kHOST);
        ITensorPtr input2 =
            createInput("input2", std::get<1>(shapes), MemoryLocation::kHOST);
        ITensorPtr output;
        EXPECT_THROW(
            {
                switch (std::get<1>(testCase))
                {
                    case Elementwise::kADD:
                        output = input1 + input2;
                        break;
                    case Elementwise::kSUB:
                        output = input1 - input2;
                        break;
                    case Elementwise::kMUL:
                        output = input1 * input2;
                        break;
                    case Elementwise::kDIV:
                        output = input1 / input2;
                        break;
                }
            },
            std::runtime_error);
    }
};

TEST_P(ElementwiseTest, testAPI) { test(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, ElementwiseTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(OPS)));

TEST_P(ElementwiseErrorTest, test) { test(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerErrorTest, ElementwiseErrorTest,
                        Combine(ValuesIn(ERROR_SHAPES), ValuesIn(OPS)));

TEST_P(ElementwiseGradientTest, testAPI) { testGradient(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, ElementwiseGradientTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(OPS)));

}  // namespace
