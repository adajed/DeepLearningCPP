#include "assign.h"
#include "graphdl_ops.h"
#include "layerTests.h"

namespace
{
using TestCase = std::tuple<UVec, MemoryLocation>;
using ErrorTestCase = std::tuple<UVec, UVec>;

std::vector<UVec> SHAPES = {
    // clang-format off
    {},
    {1},
    {1, 1},
    {1, 1, 1},
    {1, 1, 1, 1},
    {1, 1, 1, 1, 1},
    {2},
    {2, 2},
    {2, 2, 2},
    {2, 2, 2, 2},
    {2, 2, 2, 2, 2}
    // clang-format on
};

std::vector<ErrorTestCase> ERROR_SHAPES = {
    // clang-format off
    {{}, {1}},
    {{1}, {1, 1}},
    {{1, 1}, {1, 1, 1}},
    {{1, 1, 1}, {1, 1, 1, 1}},
    {{2}, {5}},
    {{2, 2}, {4}},
    {{5, 6}, {6, 5}},
    {{1, 10}, {10}}
    // clang-format on
};

class AssignTest : public LayerTest,
                   public testing::WithParamInterface<TestCase>
{
  public:
    void test(const TestCase& testCase)
    {
        UniformGen gen(seed);
        RefTensor tensor(std::get<0>(testCase), gen);

        LayerBuilder builder = [&testCase](const HostVec& ins) {
            ITensorPtr in =
                createInput("in", std::get<0>(testCase), std::get<1>(testCase));
            ITensorPtr w =
                createWeights("w", std::get<0>(testCase),
                              constantInitializer(0.), std::get<1>(testCase));
            ITensorPtr a = assign(w, in);
            initializeGraph();

            (void)a->eval({{"in", ins[0]}});
            return HostVec({w->eval({})});
        };
        bool correct = runTest({tensor}, {tensor}, builder);
        EXPECT_TRUE(correct);
    }
};

class AssignErrorTest : public LayerTest,
                        public testing::WithParamInterface<ErrorTestCase>
{
  public:
    void test(const ErrorTestCase& testCase)
    {
        ITensorPtr in =
            createInput("in", std::get<0>(testCase), MemoryLocation::kHOST);
        ITensorPtr w =
            createInput("w", std::get<1>(testCase), MemoryLocation::kHOST);
        ITensorPtr a;
        EXPECT_THROW({ a = assign(w, in); }, std::runtime_error);
    }
};

TEST_P(AssignTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TESTS(
    LayerTest, AssignTest,
    Combine(ValuesIn(SHAPES), ValuesIn(LOCATIONS))
);

TEST_P(AssignErrorTest, test)
{
    test(GetParam());
}
INSTANTIATE_TESTS(
    LayerErrorTest, AssignErrorTest,
    ValuesIn(ERROR_SHAPES)
);

}  // namespace
