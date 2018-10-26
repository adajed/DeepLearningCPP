#include "dll_errors.h"
#include "dll_ops.h"
#include "layerTests.h"
namespace
{
using TestCase = std::tuple<TensorShape, TensorShape>;

std::vector<TestCase> testCases = {
    // clang-format off
    {{1, 1}, {1, 1}},
    {{2, 2}, {2, 2}},
    {{2, 10}, {10, 3}},
    {{10, 10}, {10, 10}},
    {{1, 30}, {30, 1}},
    {{10, 100}, {100, 20}}, // big test
    // clang-format on
};

std::vector<TestCase> errorTestCases = {
    // clang-format off
    {{1, 1}, {1}},
    {{2}, {2}},
    {{2, 2}, {2, 3, 3}},
    {{10, 1}, {2, 10}}
    // clang-format on
};

class MatmulTest : public LayerTest, public testing::WithParamInterface<TestCase>
{
public:
    void test(const TestCase& testCase)
    {
        setup(testCase);

        LayerBuilder builder = [&testCase](const std::vector<HostTensor>& ins,
                                  const std::vector<HostTensor>& outs)
        {
            dll::ITensorSPtr input1 = dll::createInput("i1", std::get<0>(testCase));
            dll::ITensorSPtr input2 = dll::createInput("i2", std::get<1>(testCase));
            dll::ITensorSPtr output = dll::matmul(input1, input2);
            dll::initializeGraph();
            output->eval({{"i1", ins[0]}, {"i2", ins[1]}}, outs[0]);
        };
        bool correct = runTest({mInput1, mInput2}, {mOutput}, builder);
        EXPECT_TRUE(correct);
    }

    void testWrongShapes(const TestCase& testCase)
    {
        dll::ITensorSPtr input1 = dll::createInput("i1", std::get<0>(testCase));
        dll::ITensorSPtr input2 = dll::createInput("i2", std::get<1>(testCase));
        dll::ITensorSPtr output;
        EXPECT_THROW({
                output = dll::matmul(input1, input2);
            },
            dll::errors::NotMatchingShapesError);
    }

private:
    RefTensor mInput1, mInput2, mOutput;

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
};

TEST_P(MatmulTest, test) { test(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, MatmulTest, testing::ValuesIn(testCases));

class MatmulErrorsTest : public MatmulTest {};
TEST_P(MatmulErrorsTest, testWrongShapes) { testWrongShapes(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, MatmulErrorsTest, testing::ValuesIn(errorTestCases));

}  // namespace anonymoous
