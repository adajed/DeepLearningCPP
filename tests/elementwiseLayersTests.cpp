#include "abstractTensor.h"
#include "elementwise.h"
#include "graphdl_ops.h"
#include "layerTests.h"

namespace
{
using namespace graphdl::core::layers;

using TestCase =
    std::tuple<std::tuple<UVec, UVec>, Elementwise, MemoryLocation>;
using ErrorTestCase = std::tuple<std::tuple<UVec, UVec>, Elementwise>;

//!
//! n == 0 means to return first shape
//! n == 1 means to return second shape
//! n > 1  means to return the bigger shape
//!     (which is equal to the output shape)
//!
Shape shape(const TestCase& testCase, int n)
{
    if (n == 0)
        return std::get<0>(std::get<0>(testCase));
    else if (n == 1)
        return std::get<1>(std::get<0>(testCase));
    else
    {
        UVec v1 = shape(testCase, 0);
        UVec v2 = shape(testCase, 1);
        return v1.size() > v2.size() ? v1 : v2;
    }
}

Elementwise op(const TestCase& testCase)
{
    return std::get<1>(testCase);
}

MemoryLocation loc(const TestCase& testCase)
{
    return std::get<2>(testCase);
}

std::vector<std::tuple<UVec, UVec>> SHAPES = {
    // clang-format off
    {{}, {}},
    {{1}, {1}},
    {{1, 1}, {1, 1}},
    {{2}, {}},
    {{}, {2}},
    {{2}, {2}},
    {{2, 2}, {}},
    {{}, {2, 2}},
    {{2, 2}, {2}},
    {{2}, {2, 2}},
    {{2, 2}, {2, 2}},
    {{2, 2, 2}, {}},
    {{}, {2, 2, 2}},
    {{2, 2, 2}, {2}},
    {{2}, {2, 2, 2}},
    {{2, 2, 2}, {2, 2}},
    {{2, 2}, {2, 2, 2}},
    {{2, 2, 2}, {2, 2, 2}},
    {{2, 2, 2, 2}, {}},
    {{}, {2, 2, 2, 2}},
    {{2, 2, 2, 2}, {2, 2, 2, 2}},
    {{2, 2, 2, 2, 2}, {}},
    {{}, {2, 2, 2, 2, 2}},
    {{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}},
    {{2, 2, 2, 2, 2, 2}, {}},
    {{}, {2, 2, 2, 2, 2, 2}},
    {{2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2}},
    {{20, 20}, {}},
    /* {{}, {20, 20}}, */ // TODO: epsilon error
    {{20, 20}, {20}},
    {{20}, {20, 20}},
    {{20, 20}, {20, 20}},
    {{100, 100}, {}},
    /* {{}, {100, 100}}, */ // TODO: epsilon error
    {{100, 100}, {100, 100}},
    // clang-format on
};

std::vector<std::tuple<UVec, UVec>> ERROR_SHAPES = {
    // clang-format off
    {{2}, {5}},
    {{2, 3}, {2, 5}},
    {{3, 4}, {4, 3}},
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

std::ostream& operator<<(std::ostream& os, Elementwise e)
{
    switch (e)
    {
    case Elementwise::kADD: return os << "ADD";
    case Elementwise::kSUB: return os << "SUB";
    case Elementwise::kMUL: return os << "MUL";
    case Elementwise::kDIV: return os << "DIV";
    }

    return os;
}

class ElementwiseTest : public LayerTest,
                        public testing::WithParamInterface<TestCase>
{
  public:
    void testBack(const TestCase& testCase)
    {
        setup(testCase, true);
        LayerBuilder builder = getBuilder(testCase, true);
        bool correct = runTest({mInput1, mInput2}, {mOutput}, builder);
        EXPECT_TRUE(correct);
    }

    void testFront(const TestCase& testCase)
    {
        setup(testCase, false);
        LayerBuilder builder = getBuilder(testCase, false);
        bool correct = runTest({mInput1, mInput2}, {mOutput}, builder);
        EXPECT_TRUE(correct);
    }

    void testBackGradient(const TestCase& testCase)
    {
        setupGradient(testCase, true);
        LayerBuilder builder = getGradientBuilder(testCase, true);
        bool correct = runTest({mInput1, mInput2, mOutputGrad},
                               {mGradient1, mGradient2}, builder, 10e-2);

        EXPECT_TRUE(correct);
    }

    void testFrontGradient(const TestCase& testCase)
    {
        setupGradient(testCase, false);
        LayerBuilder builder = getGradientBuilder(testCase, false);
        bool correct = runTest({mInput1, mInput2, mOutputGrad},
                               {mGradient1, mGradient2}, builder, 10e-2);

        EXPECT_TRUE(correct);
    }

  protected:
    RefTensor mInput1, mInput2, mOutput, mOutputGrad, mGradient1, mGradient2;

  private:
    std::function<float(float, float)> getFunction(Elementwise op)
    {
        switch (op)
        {
        case Elementwise::kADD:
            return [](float f1, float f2) { return f1 + f2; };
        case Elementwise::kSUB:
            return [](float f1, float f2) { return f1 - f2; };
        case Elementwise::kMUL:
            return [](float f1, float f2) { return f1 * f2; };
        case Elementwise::kDIV:
            return [](float f1, float f2) { return f1 / f2; };
        }
        throw std::runtime_error("Unknown Elementwise");
    }

    std::function<float(float, float)> getGradient1(Elementwise op)
    {
        switch (op)
        {
        case Elementwise::kADD: return [](float f1, float f2) { return 1.; };
        case Elementwise::kSUB: return [](float f1, float f2) { return 1.; };
        case Elementwise::kMUL: return [](float f1, float f2) { return f2; };
        case Elementwise::kDIV:
            return [](float f1, float f2) { return 1. / f2; };
        }
        throw std::runtime_error("Unknown Elementwise");
    }

    std::function<float(float, float)> getGradient2(Elementwise op)
    {
        switch (op)
        {
        case Elementwise::kADD: return [](float f1, float f2) { return 1.; };
        case Elementwise::kSUB: return [](float f1, float f2) { return -1.; };
        case Elementwise::kMUL: return [](float f1, float f2) { return f1; };
        case Elementwise::kDIV:
            return [](float f1, float f2) { return -f1 / (f2 * f2); };
        }
        throw std::runtime_error("Unknown Elementwise");
    }

    void setup(const TestCase& testCase, bool back)
    {
        UniformGen gen(seed);

        // allocate memory
        mInput1 = RefTensor(shape(testCase, 0), gen);
        mInput2 = RefTensor(shape(testCase, 1), gen);
        mOutput = RefTensor(shape(testCase, 2));

        auto f = getFunction(op(testCase));

        // calculate reference output
        size_t s1 = mInput1.getCount();
        size_t s2 = mInput2.getCount();
        if (back)
        {
            for (size_t i = 0; i < (s1 > s2 ? s1 : s2); ++i)
                mOutput.at(i) = f(mInput1.at(i % s1), mInput2.at(i % s2));
        }
        else
        {
            if (s1 > s2)
            {
                size_t reduceSize = s1 / s2;
                for (size_t i = 0; i < s1; ++i)
                    mOutput.at(i) =
                        f(mInput1.at(i), mInput2.at(i / reduceSize));
            }
            else
            {
                size_t reduceSize = s2 / s1;
                for (size_t i = 0; i < s2; ++i)
                    mOutput.at(i) =
                        f(mInput1.at(i / reduceSize), mInput2.at(i));
            }
        }
    }

    void setupGradient(const TestCase& testCase, bool back)
    {
        UniformGen gen(seed);

        mInput1 = RefTensor(shape(testCase, 0), gen);
        mInput2 = RefTensor(shape(testCase, 1), gen);
        mOutputGrad = RefTensor(shape(testCase, 2), gen);
        mGradient1 = RefTensor(shape(testCase, 0));
        mGradient2 = RefTensor(shape(testCase, 1));

        auto fun1 = getGradient1(op(testCase));
        auto fun2 = getGradient2(op(testCase));

        size_t s1 = mInput1.getCount();
        size_t s2 = mInput2.getCount();
        for (size_t i = 0; i < s1; ++i) mGradient1.at(i) = 0;
        for (size_t i = 0; i < s2; ++i) mGradient2.at(i) = 0;

        if (back)
        {
            for (size_t i = 0; i < (s1 > s2 ? s1 : s2); ++i)
            {
                mGradient1.at(i % s1) +=
                    mOutputGrad.at(i) *
                    fun1(mInput1.at(i % s1), mInput2.at(i % s2));
                mGradient2.at(i % s2) +=
                    mOutputGrad.at(i) *
                    fun2(mInput1.at(i % s1), mInput2.at(i % s2));
            }
        }
        else
        {
            if (s1 > s2)
            {
                size_t reduceSize = s1 / s2;
                for (size_t i = 0; i < s1; ++i)
                {
                    mGradient1.at(i) =
                        mOutputGrad.at(i) *
                        fun1(mInput1.at(i), mInput2.at(i / reduceSize));
                    mGradient2.at(i / reduceSize) +=
                        mOutputGrad.at(i) *
                        fun2(mInput1.at(i), mInput2.at(i / reduceSize));
                }
            }
            else
            {
                size_t reduceSize = s2 / s1;
                for (size_t i = 0; i < s2; ++i)
                {
                    mGradient1.at(i / reduceSize) +=
                        mOutputGrad.at(i) *
                        fun1(mInput1.at(i / reduceSize), mInput2.at(i));
                    mGradient2.at(i) =
                        mOutputGrad.at(i) *
                        fun2(mInput1.at(i / reduceSize), mInput2.at(i));
                }
            }
        }
    }

    LayerBuilder getBuilder(const TestCase& testCase, bool back)
    {
        return [&testCase, back](const HostVec& ins) {
            MemoryType type = memoryLocationToType(loc(testCase));
            core::Graph::SPtr graph = core::getDefaultGraph();

            Tensor::SPtr in1 = graph->addInput(
                "in1",
                createLayer<InputLayer>("in1", shape(testCase, 0), type));
            Tensor::SPtr in2 = graph->addInput(
                "in2",
                createLayer<InputLayer>("in2", shape(testCase, 1), type));

            Tensor::SPtr output;
            if (back)
                output = elementwiseBack(in1, in2, op(testCase));
            else
                output = elementwiseFront(in1, in2, op(testCase));

            ITensorPtr t = makeAbstractTensor(output);

            initializeGraph();

            return HostVec({t->eval({{"in1", ins[0]}, {"in2", ins[1]}})});
        };
    }

    LayerBuilder getGradientBuilder(const TestCase& testCase, bool back)
    {
        return [&testCase, back](const HostVec& ins) {
            MemoryType type = memoryLocationToType(std::get<2>(testCase));
            Tensor::SPtr in1 = core::getDefaultGraph()->addInput(
                "in1",
                createLayer<InputLayer>("in1", shape(testCase, 0), type));
            Tensor::SPtr in2 = core::getDefaultGraph()->addInput(
                "in2",
                createLayer<InputLayer>("in2", shape(testCase, 1), type));
            Tensor::SPtr outG = core::getDefaultGraph()->addInput(
                "outG",
                createLayer<InputLayer>("outG", shape(testCase, 2), type));

            Tensor::SPtr output;
            Layer::SPtr layer;

            if (back)
            {
                output = elementwiseBack(in1, in2, op(testCase));
                layer = createLayer<ElementwiseBackGradientLayer>(
                    in1, in2, output, outG, op(testCase));
            }
            else
            {
                output = elementwiseFront(in1, in2, op(testCase));
                layer = createLayer<ElementwiseFrontGradientLayer>(
                    in1, in2, output, outG, op(testCase));
            }
            initializeGraph();
            std::vector<Tensor::SPtr> grads = layer->getOutputs();

            std::vector<ITensorPtr> igrads = {makeAbstractTensor(grads[0]),
                                              makeAbstractTensor(grads[1])};
            return eval(igrads,
                        {{"in1", ins[0]}, {"in2", ins[1]}, {"outG", ins[2]}});
        };
    }
};

class ElementwiseBackTest : public ElementwiseTest
{
};
class ElementwiseFrontTest : public ElementwiseTest
{
};
class ElementwiseBackGradientTest : public ElementwiseTest
{
};
class ElementwiseFrontGradientTest : public ElementwiseTest
{
};

class ElementwiseErrorTest : public LayerTest,
                             public testing::WithParamInterface<ErrorTestCase>
{
  public:
    void test(const ErrorTestCase& testCase)
    {
        std::tuple<UVec, UVec> shapes = std::get<0>(testCase);
        ITensorPtr input1 =
            createInput("input1", std::get<0>(shapes), MemoryLocation::kHOST);
        ITensorPtr input2 =
            createInput("input2", std::get<1>(shapes), MemoryLocation::kHOST);
        ITensorPtr output;
        EXPECT_THROW(
            {
                switch (std::get<1>(testCase))
                {
                case Elementwise::kADD: output = input1 + input2; break;
                case Elementwise::kSUB: output = input1 - input2; break;
                case Elementwise::kMUL: output = input1 * input2; break;
                case Elementwise::kDIV: output = input1 / input2; break;
                }
            },
            std::runtime_error);
    }
};

TEST_P(ElementwiseBackTest, testAPI)
{
    testBack(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, ElementwiseBackTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(OPS),
                                ValuesIn(LOCATIONS)));

TEST_P(ElementwiseFrontTest, testAPI)
{
    testFront(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, ElementwiseFrontTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(OPS),
                                ValuesIn(LOCATIONS)));

TEST_P(ElementwiseBackGradientTest, testAPI)
{
    testBackGradient(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, ElementwiseBackGradientTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(OPS),
                                ValuesIn(LOCATIONS)));

TEST_P(ElementwiseFrontGradientTest, testAPI)
{
    testFrontGradient(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, ElementwiseFrontGradientTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(OPS),
                                ValuesIn(LOCATIONS)));

TEST_P(ElementwiseErrorTest, test)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerErrorTest, ElementwiseErrorTest,
                        Combine(ValuesIn(ERROR_SHAPES), ValuesIn(OPS)));

}  // namespace
