#include "graphdl_ops.h"
#include "group.h"
#include "layerTests.h"

namespace
{
using TestCase = std::tuple<int, MemoryLocation>;

class GroupTest : public LayerTest, public testing::WithParamInterface<TestCase>
{
  public:
    void test(const TestCase& testCase)
    {
        int n = std::get<0>(testCase);
        UniformGen gen(seed);

        std::vector<RefTensor> inputs;
        for (int i = 0; i < n; ++i)
            inputs.push_back(RefTensor(std::vector<unsigned>({}), gen));

        LayerBuilder builder = [&testCase](const HostVec& ins) {
            InputDict inputs;
            std::vector<ITensorPtr> assigns;
            std::vector<ITensorPtr> weights;
            for (int i = 0; i < std::get<0>(testCase); ++i)
            {
                std::string iName = "input" + std::to_string(i);
                std::string wName = "weights" + std::to_string(i);
                ITensorPtr iTensor =
                    createInput(iName, {}, std::get<1>(testCase));
                ITensorPtr wTensor = createWeights(
                    wName, {}, constantInitializer(0.), std::get<1>(testCase));
                assigns.push_back(assign(wTensor, iTensor));
                weights.push_back(wTensor);
                inputs.insert({iName, ins[i]});
            }

            ITensorPtr g = group(assigns);
            initializeGraph();

            (void)g->eval(inputs);
            return eval(weights, {});
        };

        bool correct = runTest(inputs, inputs, builder);
        EXPECT_TRUE(correct);
    }
};

TEST_P(GroupTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TESTS(
    LayerTest, GroupTest,
    Combine(Range(1, 10), ValuesIn(LOCATIONS))
);

}  // namespace
