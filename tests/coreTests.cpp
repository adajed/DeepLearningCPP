#include <gtest/gtest.h>
#include "dll.h"
#include "dll_errors.h"
#include "dll_ops.h"
#include "graph.h"

#include <random>

#define COMMA ,

std::random_device rd;
std::mt19937 e2(rd());
std::uniform_real_distribution<> dist(-5., 5.);

class CoreTest : public testing::Test
{
   protected:
    void SetUp() override { testing::Test::SetUp(); }

    void TearDown() override
    {
        dll::core::GraphRegister::getGlobalGraphRegister().clear();
        testing::Test::TearDown();
    }
};

const std::string SAMPLE_NAME = "sample_name";

TEST_F(CoreTest, simple)
{
    dll::IGraphSPtr g = dll::createGraph(SAMPLE_NAME);
    EXPECT_NE(g, nullptr);
}

TEST_F(CoreTest, graphWithGivenNameAlreadyExists)
{
    dll::IGraphSPtr g = dll::createGraph(SAMPLE_NAME);
    EXPECT_NE(g.get(), nullptr);
    g = dll::createGraph(SAMPLE_NAME);
    EXPECT_EQ(g.get(), nullptr);
}

TEST_F(CoreTest, setDefaultGraph)
{
    dll::IGraphSPtr g = dll::createGraph(SAMPLE_NAME);
    dll::setDefaultGraph(g);
    dll::IGraphSPtr g2 = dll::getDefaultGraph();
    EXPECT_EQ(g, g2);
}

TEST_F(CoreTest, emptyInput)
{
    auto inputs = dll::getDefaultGraph()->getInputs();
    EXPECT_EQ(inputs.size(), 0);
}

TEST_F(CoreTest, addInput)
{
    const std::string INPUT_NAME = "input1";
    dll::ITensorSPtr input = dll::createInput(INPUT_NAME, {3, 224, 224});
    auto inputs = dll::getDefaultGraph()->getInputs();
    std::map<std::string, dll::ITensorSPtr> testMap = {{INPUT_NAME, input}};
    EXPECT_EQ(inputs, testMap);
}

TEST_F(CoreTest, emptyWeights)
{
    auto weights = dll::getDefaultGraph()->getWeights();
    EXPECT_EQ(weights.size(), 0);
}

TEST_F(CoreTest, addWeights)
{
    const std::string WEIGHTS_NAME = "weights";
    dll::ITensorSPtr w = dll::createWeights(WEIGHTS_NAME, {100, 100});
    auto weights = dll::getDefaultGraph()->getWeights();
    std::map<std::string, dll::ITensorSPtr> testMap = {{WEIGHTS_NAME, w}};
    EXPECT_EQ(weights, testMap);
}

TEST_F(CoreTest, addInputWithTheSameName)
{
    dll::ITensorSPtr input1 = dll::createInput("input1", {3, 224, 224});
    EXPECT_NE(input1, nullptr);
    dll::ITensorSPtr input2 = dll::createInput("input1", {3, 224, 224});
    EXPECT_EQ(input2, nullptr);
}

TEST_F(CoreTest, gradients)
{
    dll::ITensorSPtr i = dll::createInput("input", {1});
    dll::ITensorSPtr w = dll::createWeights("weights", {1});
    dll::ITensorSPtr output = (dll::constant(1., {1}) / i) * w;
    dll::ITensorSPtr grad = dll::gradients(output)[w];
    dll::initializeGraph();

    dll::HostTensor iH{nullptr, 1};
    dll::HostTensor wH{nullptr, 1};
    dll::HostTensor gH{nullptr, 1};
    iH.values = new float[1];
    wH.values = new float[1];
    gH.values = new float[1];
    iH.values[0] = 5.;

    dll::eval({w, grad}, {{"input", iH}}, {wH, gH});
    EXPECT_FLOAT_EQ(gH.values[0], 1. / 5.);

    delete[] iH.values;
    delete[] wH.values;
    delete[] gH.values;
}

TEST_F(CoreTest, nonScalarGradientException)
{
    dll::ITensorSPtr i = dll::createInput("input", {2, 2});
    dll::ITensorSPtr w = dll::createWeights("weights", {2, 2});
    dll::ITensorSPtr o = i * w;
    dll::ITensorSPtr grad;
    EXPECT_THROW({ grad = dll::gradients(o)[w]; },
                 dll::errors::NotScalarGradientsCalculation);
}
