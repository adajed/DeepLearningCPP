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
        UniformGen gen(0);

        std::vector<RefTensor> inputs;
        for (int i = 0; i < n; ++i)
        {
            RefTensor t(std::vector<unsigned>({}));
            t.fillRandomly(gen);
            inputs.push_back(t);
        }

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
INSTANTIATE_TEST_CASE_P(LayerTest, GroupTest,
                        Combine(Range(1, 10), ValuesIn(LOCATIONS)));

}  // namespace
