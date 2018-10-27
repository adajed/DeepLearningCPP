#include "elementwiseOper.h"
#include "layerTests.h"

#include "dll_errors.h"
#include "dll_ops.h"

namespace
{
using Elementwise = dll::core::layers::Elementwise;
using Vec = std::vector<unsigned>;
using TestCase = std::tuple<Vec, Elementwise>;
using ErrorTestCase = std::tuple<std::tuple<Vec, Vec>, Elementwise>;

std::vector<Vec> testCases = {
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

std::vector<std::tuple<Vec, Vec>> errorTestCases = {
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

std::vector<Elementwise> testOps = {
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

   protected:
    RefTensor mInput1, mInput2, mOutput;

   private:
    void setup(const TestCase& testCase)
    {
        UniformGen gen(0);

        // allocate memory
        mInput1 = RefTensor(std::get<0>(testCase));
        mInput2 = RefTensor(std::get<0>(testCase));
        mOutput = RefTensor(std::get<0>(testCase));

        // fill inputs with random data
        mInput1.fillRandomly(gen);
        mInput2.fillRandomly(gen);

        std::function<float(float, float)> f;
        switch (std::get<1>(testCase))
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
        for (std::size_t i = 0; i < mInput1.count(); ++i)
            mOutput.at(i) = f(mInput1.at(i), mInput2.at(i));
    }

    LayerBuilder getBuilder(const TestCase& testCase)
    {
        return [testCase](const std::vector<HostTensor>& ins,
                          const std::vector<HostTensor>& outs) {
            dll::ITensorSPtr input1 =
                dll::createInput("input1", std::get<0>(testCase));
            dll::ITensorSPtr input2 =
                dll::createInput("input2", std::get<0>(testCase));
            dll::ITensorSPtr output;
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
            dll::initializeGraph();
            output->eval({{"input1", ins[0]}, {"input2", ins[1]}}, outs[0]);
        };
    }
};

class ElementwiseErrorTest : public LayerTest,
                             public testing::WithParamInterface<ErrorTestCase>
{
   public:
    void test(const ErrorTestCase& testCase)
    {
        std::tuple<Vec, Vec> shapes = std::get<0>(testCase);
        dll::ITensorSPtr input1 =
            dll::createInput("input1", std::get<0>(shapes));
        dll::ITensorSPtr input2 =
            dll::createInput("input2", std::get<1>(shapes));
        dll::ITensorSPtr output;
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
            dll::errors::NotMatchingShapesError);
    }
};

TEST_P(ElementwiseTest, test) { test(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, ElementwiseTest,
                        testing::Combine(testing::ValuesIn(testCases),
                                         testing::ValuesIn(testOps)));

TEST_P(ElementwiseErrorTest, test) { test(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerErrorTest, ElementwiseErrorTest,
                        testing::Combine(testing::ValuesIn(errorTestCases),
                                         testing::ValuesIn(testOps)));

}  // namespace
