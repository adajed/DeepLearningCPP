#include "graphdl_ops.h"
#include "layerTests.h"
#include "reshape.h"

namespace
{
using Shapes = std::tuple<UVec, UVec>;
using TestCase = std::tuple<Shapes, MemoryLocation>;

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

UVec shape0(TestCase testCase)
{
    return std::get<0>(std::get<0>(testCase));
}

UVec shape1(TestCase testCase)
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
        UniformGen gen(seed);
        RefTensor input = RefTensor(shape0(testCase), gen);
        RefTensor output = RefTensor(shape1(testCase));

        for (size_t pos = 0; pos < input.getCount(); ++pos)
            output.at(pos) = input.at(pos);

        LayerBuilder builder = [&testCase](const HostVec& ins) {
            ITensorPtr in = createInput("in", shape0(testCase), loc(testCase));
            ITensorPtr out = reshape(in, shape1(testCase));
            initializeGraph();
            return HostVec({out->eval({{"in", ins[0]}})});
        };

        bool correct = runTest({input}, {output}, builder);
        EXPECT_TRUE(correct);
    }
};

class ReshapeErrorTest : public LayerTest,
                         public testing::WithParamInterface<TestCase>
{
  public:
    void test(const TestCase& testCase)
    {
        ITensorPtr in = createInput("in", shape0(testCase), loc(testCase));
        EXPECT_THROW({ ITensorPtr t = reshape(in, shape1(testCase)); },
                     std::runtime_error);
    }
};

TEST_P(ReshapeTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, ReshapeTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(LOCATIONS)));

TEST_P(ReshapeErrorTest, test)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerErrorTest, ReshapeErrorTest,
                        Combine(ValuesIn(ERROR_SHAPES), ValuesIn(LOCATIONS)));

}  // namespace
