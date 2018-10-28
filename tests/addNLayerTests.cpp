#include "addN.h"
#include "dll_ops.h"
#include "layerTests.h"

namespace
{
using namespace dll::core::layers;
using TestCase = std::tuple<int, Vec>;

std::vector<Vec> SHAPES = {
    // clang-format off
    {},
    {1},
    {1, 1},
    {1, 1, 1},
    {1, 1, 1, 1},
    {2},
    {2, 2},
    {2, 2, 2},
    {2, 2, 2, 2},
    {10},
    {1, 10},
    {5, 13}
    // clang-format on
};

class AddNTest : public LayerTest, public testing::WithParamInterface<TestCase>
{
   public:
    void test(const TestCase& testCase)
    {
        UniformGen gen(0);
        std::vector<RefTensor> inputs;
        RefTensor output(std::get<1>(testCase));
        for (int i = 0; i < std::get<0>(testCase); ++i)
        {
            inputs.push_back(RefTensor(std::get<1>(testCase)));
            inputs[i].fillRandomly(gen);
        }

        for (std::size_t pos = 0; pos < output.count(); ++pos)
        {
            output.at(pos) = 0.;
            for (int i = 0; i < std::get<0>(testCase); ++i)
                output.at(pos) += inputs[i].at(pos);
        }

        LayerBuilder builder = [&testCase](const HostVec& ins,
                                           const HostVec& outs) {
            std::vector<ITensorSPtr> inputs;
            std::map<std::string, HostTensor> inMap;
            for (int i = 0; i < std::get<0>(testCase); ++i)
            {
                std::string name = "i" + std::to_string(i);
                inputs.push_back(createInput(name, std::get<1>(testCase)));
                inMap.insert({name, ins[i]});
            }
            ITensorSPtr output = addN(inputs);
            initializeGraph();
            output->eval(inMap, outs[0]);
        };
        bool correct = runTest(inputs, {output}, builder);
        EXPECT_TRUE(correct);
    }
};

TEST_P(AddNTest, testAPI) { test(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, AddNTest,
                        Combine(Range(1, 11), ValuesIn(SHAPES)));

}  // namespace
