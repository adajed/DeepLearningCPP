#include "graphdl_ops.h"
#include "layerTests.h"
#include "queue.h"

namespace
{
using TestCase = std::tuple<int, MemoryLocation>;

class QueueTest : public LayerTest, public testing::WithParamInterface<TestCase>
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
            std::vector<ITensorPtr> ops;
            std::vector<ITensorPtr> weights;
            for (int i = 0; i < std::get<0>(testCase); ++i)
            {
                std::string wName = "weights" + std::to_string(i);
                weights.push_back(createWeights(
                    wName, {}, constantInitializer(0.), std::get<1>(testCase)));
            }
            for (int i = 0; i < std::get<0>(testCase); ++i)
            {
                std::string iName = "input" + std::to_string(i);
                ITensorPtr iTensor =
                    createInput(iName, {}, std::get<1>(testCase));

                std::vector<ITensorPtr> assigns;
                for (int j = i; j < std::get<0>(testCase); ++j)
                    assigns.push_back(assign(weights[j], iTensor));
                ops.push_back(group(assigns));
                inputs.insert({iName, ins[i]});
            }

            ITensorPtr q = queue(ops);
            initializeGraph();

            (void)q->eval(inputs);
            return eval(weights, {});
        };

        bool correct = runTest(inputs, inputs, builder);
        EXPECT_TRUE(correct);
    }
};

TEST_P(QueueTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, QueueTest,
                        Combine(Range(1, 10), ValuesIn(LOCATIONS)));

}  // namespace
