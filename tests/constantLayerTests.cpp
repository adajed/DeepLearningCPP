#include "constant.h"
#include "graphdl_ops.h"
#include "layerTests.h"

namespace
{
using TestCase = std::tuple<UVec, float, MemoryLocation>;

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

class ConstantTest : public LayerTest,
                     public testing::WithParamInterface<TestCase>
{
  public:
    void test(const TestCase& testCase)
    {
        RefTensor tensor(std::get<0>(testCase));
        for (std::size_t pos = 0; pos < tensor.getCount(); ++pos)
            tensor.at(pos) = std::get<1>(testCase);

        LayerBuilder builder = [&testCase](const HostVec& ins) {
            ITensorPtr c =
                constant(std::get<1>(testCase), std::get<0>(testCase),
                         std::get<2>(testCase));
            initializeGraph();

            return HostVec({c->eval({})});
        };
        bool correct = runTest({}, {tensor}, builder);
        EXPECT_TRUE(correct);
    }
};

TEST_P(ConstantTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TESTS(
    LayerTest, ConstantTest,
    Combine(ValuesIn(SHAPES), ValuesIn({3.14f}), ValuesIn(LOCATIONS))
);

}  // namespace
