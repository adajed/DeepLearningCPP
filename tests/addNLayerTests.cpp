#include "abstractTensor.h"
#include "addN.h"
#include "graph.h"
#include "graphdl_ops.h"
#include "layerTests.h"

namespace
{
using namespace graphdl::core::layers;
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

        for (std::size_t pos = 0; pos < output.getCount(); ++pos)
        {
            output.at(pos) = 0.;
            for (int i = 0; i < std::get<0>(testCase); ++i)
                output.at(pos) += inputs[i].at(pos);
        }

        LayerBuilder builder = [&testCase](const HostVec& ins) {
            std::vector<ITensorPtr> inputs;
            std::map<std::string, HostTensor> inMap;
            for (int i = 0; i < std::get<0>(testCase); ++i)
            {
                std::string name = "i" + std::to_string(i);
                inputs.push_back(createInput(name, std::get<1>(testCase)));
                inMap.insert({name, ins[i]});
            }
            ITensorPtr output = addN(inputs);
            initializeGraph();
            return HostVec({output->eval(inMap)});
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

        for (std::size_t pos = 0; pos < outputGrad.getCount(); ++pos)
        {
            for (int i = 0; i < std::get<0>(testCase); ++i)
                inputGrads[i].at(pos) = outputGrad.at(pos);
        }
        inputs.push_back(outputGrad);

        LayerBuilder builder = [&testCase](const HostVec& ins) {
            std::vector<Tensor::SPtr> inputs;
            std::map<std::string, HostTensor> inMap;
            for (int i = 0; i < std::get<0>(testCase); ++i)
            {
                std::string name = "i" + std::to_string(i);
                Layer::SPtr inputLayer =
                    createLayer<InputLayer>(name, std::get<1>(testCase));
                inputs.push_back(
                    core::getDefaultGraph()->addInput(name, inputLayer));
                inMap.insert({name, ins[i]});
            }
            Layer::SPtr outputGradLayer =
                createLayer<InputLayer>("outG", std::get<1>(testCase));
            Tensor::SPtr outputGrad =
                core::getDefaultGraph()->addInput("outG", outputGradLayer);
            inMap.insert({"outG", ins.back()});
            Tensor::SPtr output = core::addN(inputs);
            Layer::SPtr gradLayer =
                createLayer<AddNGradientLayer>(inputs, output, outputGrad);
            std::vector<Tensor::SPtr> inputGrads = gradLayer->getOutputs();
            std::vector<ITensorPtr> calcTensors;
            for (Tensor::SPtr t : inputGrads)
                calcTensors.push_back(makeAbstractTensor(t));
            initializeGraph();

            return eval(calcTensors, inMap);
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
        std::vector<ITensorPtr> inputs;
        for (unsigned i = 0; i < testCase.size(); ++i)
        {
            std::string name = "i" + std::to_string(i);
            inputs.push_back(createInput(name, testCase[i]));
        }
        ITensorPtr output;
        EXPECT_THROW({ output = addN(inputs); }, std::runtime_error);
    }
};

class AddNGradientTest : public AddNTest
{
};

TEST_P(AddNTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, AddNTest,
                        Combine(Range(1, 11), ValuesIn(SHAPES)));

TEST_P(AddNGradientTest, testAPI)
{
    testGradient(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, AddNGradientTest,
                        Combine(Range(1, 11), ValuesIn(SHAPES)));

TEST_P(AddNErrorTest, test)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerErrorTest, AddNErrorTest, ValuesIn(ERROR_SHAPES));

}  // namespace
