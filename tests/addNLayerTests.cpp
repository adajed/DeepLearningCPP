#include "addN.h"
#include "dll_ops.h"
#include "graph.h"
#include "layerTests.h"

namespace
{
using namespace dll::core::layers;
using TestCase = std::tuple<int, Vec>;
using ErrorTestCase = std::vector<Vec>;

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

std::vector<std::vector<Vec>> ERROR_SHAPES = {
    // clang-format off
    {},
    {{}, {1}},
    {{}, {}, {1}},
    {{1}, {1, 1}},
    {{2}, {4}},
    {{2, 2}, {4}},
    {{5}, {5}, {5}, {5}, {4}, {5}, {5}}
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

    void testGradient(const TestCase& testCase)
    {
        UniformGen gen(0);
        std::vector<RefTensor> inputs;
        std::vector<RefTensor> inputGrads;
        for (int i = 0; i < std::get<0>(testCase); ++i)
        {
            inputs.push_back(RefTensor(std::get<1>(testCase)));
            inputs[i].fillRandomly(gen);
            inputGrads.push_back(RefTensor(std::get<1>(testCase)));
        }
        RefTensor outputGrad(std::get<1>(testCase));
        outputGrad.fillRandomly(gen);

        for (std::size_t pos = 0; pos < outputGrad.count(); ++pos)
        {
            for (int i = 0; i < std::get<0>(testCase); ++i)
                inputGrads[i].at(pos) = outputGrad.at(pos);
        }
        inputs.push_back(outputGrad);

        LayerBuilder builder = [&testCase](const HostVec& ins,
                                           const HostVec& outs) {
            std::vector<Tensor::SPtr> inputs;
            std::map<std::string, HostTensor> inMap;
            for (int i = 0; i < std::get<0>(testCase); ++i)
            {
                std::string name = "i" + std::to_string(i);
                inputs.push_back(core::getDefaultGraph()->addInput(
                    name, std::get<1>(testCase)));
                inMap.insert({name, ins[i]});
            }
            Tensor::SPtr outputGrad = core::getDefaultGraph()->addInput(
                "outG", std::get<1>(testCase));
            inMap.insert({"outG", ins.back()});
            Tensor::SPtr output = core::addN(inputs);
            Oper::SPtr oper =
                std::make_shared<AddNGradientOper>(inputs, output, outputGrad);
            core::getDefaultGraph()->insertOperation(oper);
            std::vector<Tensor::SPtr> inputGrads = oper->getOutputs();
            std::vector<ITensorSPtr> calcTensors;
            for (Tensor::SPtr t : inputGrads)
                calcTensors.push_back(ITensorSPtr(t));
            initializeGraph();

            eval(calcTensors, inMap, outs);
        };
        bool correct = runTest(inputs, inputGrads, builder);
        EXPECT_TRUE(correct);
    }
};

class AddNErrorTest : public LayerTest,
                      public testing::WithParamInterface<ErrorTestCase>
{
   public:
    void test(const ErrorTestCase& testCase)
    {
        std::vector<ITensorSPtr> inputs;
        for (unsigned i = 0; i < testCase.size(); ++i)
        {
            std::string name = "i" + std::to_string(i);
            inputs.push_back(createInput(name, testCase[i]));
        }
        ITensorSPtr output;
        EXPECT_THROW({ output = addN(inputs); }, std::invalid_argument);
    }
};

class AddNGradientTest : public AddNTest
{
};

TEST_P(AddNTest, testAPI) { test(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, AddNTest,
                        Combine(Range(1, 11), ValuesIn(SHAPES)));

TEST_P(AddNGradientTest, testAPI) { testGradient(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerTest, AddNGradientTest,
                        Combine(Range(1, 11), ValuesIn(SHAPES)));

TEST_P(AddNErrorTest, test) { test(GetParam()); }
INSTANTIATE_TEST_CASE_P(LayerErrorTest, AddNErrorTest, ValuesIn(ERROR_SHAPES));

}  // namespace
