#include "reshape.h"
#include "graphdl_ops.h"
#include "layerTests.h"

namespace
{
using Shapes = std::tuple<Vec, Vec>;
using TestCase = std::tuple<Shapes, MemoryLocation>;
using ErrorTestCase = std::tuple<Shapes, MemoryLocation>;

std::vector<Shapes> SHAPES = {
    // clang-format off
    {{}, {}},
    {{}, {1}},
    {{1}, {}},
    {{}, {1, 1}},
    {{5}, {1, 5}},
    {{1, 5}, {5}},
    {{12}, {1, 3, 4}},
    {{12}, {2, 1, 3, 1, 2, 1}},
    {{2, 1, 3, 1, 2, 1}, {12}}
    // clang-format on
};

std::vector<Shapes> ERROR_SHAPES = {
    // clang-format off
    {{1}, {2}},
    {{2}, {0, 2}}
    // clang-format on
};

Vec shape0(TestCase testCase)
{
    return std::get<0>(std::get<0>(testCase));
}

Vec shape1(TestCase testCase)
{
    return std::get<1>(std::get<0>(testCase));
}

MemoryLocation loc(TestCase testCase)
{
    return std::get<1>(testCase);
}

class ReshapeTest : public LayerTest,
                    public testing::WithParamInterface<TestCase>
{
  public:
    void test(const TestCase& testCase)
    {
        UniformGen gen(0);
        RefTensor input = RefTensor(shape0(testCase));
        RefTensor output = RefTensor(shape1(testCase));

        for (size_t pos = 0; pos < input.getCount(); ++pos)
            output.at(pos) = input.at(pos);

        LayerBuilder builder = [&testCase](const HostVec& ins)
        {
            ITensorPtr in = createInput("in", shape0(testCase), loc(testCase));
            ITensorPtr out = reshape(in, shape1(testCase));
            initializeGraph();
            return HostVec({out->eval({{"in", ins[0]}})});
        };

        bool correct = runTest({input}, {output}, builder);
        EXPECT_TRUE(correct);
    }
};

TEST_P(ReshapeTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, ReshapeTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(LOCATIONS)));

}
