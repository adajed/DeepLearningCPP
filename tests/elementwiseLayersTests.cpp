#include "layerTests.h"

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

        // calculate reference output
        for (std::size_t i = 0; i < mInput1.count(); ++i)
            mOutput.at(i) = elementwise(mInput1.at(i), mInput2.at(i));
    }

    virtual float elementwise(float f1, float f2) = 0;

    virtual LayerBuilder getBuilder(TestCase testCase) = 0;
};

#define TEST_CLASS(ClassName, op)                                           \
    class ClassName : public ElementwiseTest                                \
    {                                                                       \
       private:                                                             \
        float elementwise(float f1, float f2) override { return f1 op f2; } \
                                                                            \
        LayerBuilder getBuilder(TestCase testCase) override                 \
        {                                                                   \
            return [testCase](const std::vector<HostTensor>& ins,           \
                              const std::vector<HostTensor>& outs) {        \
                dll::ITensorSPtr input1 =                                   \
                    dll::createInput("input1", std::get<0>(testCase));      \
                dll::ITensorSPtr input2 =                                   \
                    dll::createInput("input2", std::get<0>(testCase));      \
                dll::ITensorSPtr output = input1 op input2;                 \
                dll::initializeGraph();                                     \
                                                                            \
                output->eval({{"input1", ins[0]}, {"input2", ins[1]}},      \
                             outs[0]);                                      \
            };                                                              \
        }                                                                   \
    };                                                                      \
    TEST_P(ClassName, test) { test(GetParam()); }                           \
    INSTANTIATE_TEST_CASE_P(ClassName##Instantiation, ClassName,            \
                            testing::ValuesIn(testCases))

TEST_CLASS(AddTest, +);
TEST_CLASS(SubTest, -);
TEST_CLASS(MulTest, *);
TEST_CLASS(DivTest, /);

}  // namespace
