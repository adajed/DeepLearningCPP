#include "layerTests.h"

#include "dll_errors.h"
#include "dll_ops.h"

namespace
{
using TestCase = std::tuple<TensorShape>;

std::vector<TestCase> testCases = {
    // clang-format off
    {{1}},
    {{1, 1}},
    {{2}},
    {{2, 2}},
    {{2, 2, 2}}
    // clang-format on
};

enum class Elementwise
{
    kADD = 0,
    kSUB = 1,
    kMUL = 2,
    kDIV = 3
};

template <Elementwise oper>
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

    void testShapeException()
    {
        dll::ITensorSPtr input1 = dll::createInput("input1", {2});
        dll::ITensorSPtr input2 = dll::createInput("input2", {2, 2});
        dll::ITensorSPtr output;
        EXPECT_THROW(
            {
                switch (oper)
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

        // calculate reference output
        for (std::size_t i = 0; i < mInput1.count(); ++i)
            mOutput.at(i) = elementwise(mInput1.at(i), mInput2.at(i));
    }

    float elementwise(float f1, float f2)
    {
        switch (oper)
        {
            case Elementwise::kADD:
                return f1 + f2;
            case Elementwise::kSUB:
                return f1 - f2;
            case Elementwise::kMUL:
                return f1 * f2;
            case Elementwise::kDIV:
                return f1 / f2;
        }
    }

    LayerBuilder getBuilder(TestCase testCase)
    {
        return [testCase](const std::vector<HostTensor>& ins,
                          const std::vector<HostTensor>& outs) {
            dll::ITensorSPtr input1 =
                dll::createInput("input1", std::get<0>(testCase));
            dll::ITensorSPtr input2 =
                dll::createInput("input2", std::get<0>(testCase));
            dll::ITensorSPtr output;
            switch (oper)
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

#define TEST_ELEMENTWISE(ClassName)                                 \
    TEST_F(ClassName, testShapeException) { testShapeException(); } \
    TEST_P(ClassName, test) { test(GetParam()); }                   \
    INSTANTIATE_TEST_CASE_P(LayerTest, ClassName, testing::ValuesIn(testCases))

class AddTest : public ElementwiseTest<Elementwise::kADD>
{
};
TEST_ELEMENTWISE(AddTest);

class SubTest : public ElementwiseTest<Elementwise::kSUB>
{
};
TEST_ELEMENTWISE(SubTest);

class MulTest : public ElementwiseTest<Elementwise::kMUL>
{
};
TEST_ELEMENTWISE(MulTest);

class DivTest : public ElementwiseTest<Elementwise::kDIV>
{
};
TEST_ELEMENTWISE(DivTest);

}  // namespace
