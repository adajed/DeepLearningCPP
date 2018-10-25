#include "layerTests.h"

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

class InputTest : public LayerTest, public testing::WithParamInterface<TestCase>
{
   public:
    void test(const TestCase& testCase)
    {
        setup(testCase);

        LayerBuilder builder = [testCase](const std::vector<HostTensor>& ins,
                                          const std::vector<HostTensor>& outs) {
            dll::ITensorSPtr input =
                dll::createInput("input", std::get<0>(testCase));
            dll::initializeGraph();

            input->eval({{"input", ins[0]}}, outs[0]);
        };
        bool correct = runTest({mInput}, {mOutput}, builder);
        EXPECT_TRUE(correct);
    }

   private:
    RefTensor mInput, mOutput;

    void setup(const TestCase& testCase)
    {
        UniformGen gen(0);

        mInput = RefTensor(std::get<0>(testCase));
        mOutput = RefTensor(std::get<0>(testCase));

        mInput.fillRandomly(gen);

        // calculate reference output
        for (std::size_t i = 0; i < mInput.count(); ++i)
            mOutput.at(i) = mInput.at(i);
    }
};

TEST_P(InputTest, test) { test(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, InputTest, testing::ValuesIn(testCases));
}  // namespace
