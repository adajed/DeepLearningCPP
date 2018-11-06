#include "graph.h"
#include "graphdl.h"
#include "graphdl_ops.h"

#include <gtest/gtest.h>
#include <random>

std::random_device rd;
std::mt19937 e2(rd());
std::uniform_real_distribution<> dist(-5., 5.);

class CoreTest : public testing::Test
{
  protected:
    void SetUp() override { testing::Test::SetUp(); }

    void TearDown() override
    {
        graphdl::core::GraphRegister::getGlobalGraphRegister().clear();
        testing::Test::TearDown();
    }
};

const std::string SAMPLE_NAME = "sample_name";

TEST_F(CoreTest, simple)
{
    graphdl::IGraphPtr g = graphdl::createGraph(SAMPLE_NAME);
    EXPECT_NE(g, nullptr);
}

TEST_F(CoreTest, graphWithGivenNameAlreadyExists)
{
    graphdl::IGraphPtr g = graphdl::createGraph(SAMPLE_NAME);
    EXPECT_NE(g.get(), nullptr);
    EXPECT_THROW({ g = graphdl::createGraph(SAMPLE_NAME); },
                 std::runtime_error);
}

TEST_F(CoreTest, setDefaultGraph)
{
    graphdl::IGraphPtr g = graphdl::createGraph(SAMPLE_NAME);
    graphdl::setDefaultGraph(g);
    graphdl::IGraphPtr g2 = graphdl::getDefaultGraph();
    EXPECT_EQ(g->getName(), g2->getName());
}

TEST_F(CoreTest, emptyInput)
{
    auto inputs = graphdl::getDefaultGraph()->getInputs();
    EXPECT_EQ(inputs.size(), 0);
}

TEST_F(CoreTest, addInput)
{
    const std::string INPUT_NAME = "input1";
    graphdl::ITensorPtr input = graphdl::createInput(INPUT_NAME, {3, 224, 224});
    auto inputs = graphdl::getDefaultGraph()->getInputs();
    EXPECT_EQ(inputs.size(), 1);
    EXPECT_EQ(inputs.count(INPUT_NAME), 1);
}

TEST_F(CoreTest, emptyWeights)
{
    auto weights = graphdl::getDefaultGraph()->getWeights();
    EXPECT_EQ(weights.size(), 0);
}

TEST_F(CoreTest, addWeights)
{
    const std::string WEIGHTS_NAME = "weights";
    graphdl::ITensorPtr w = graphdl::createWeights(WEIGHTS_NAME, {100, 100});
    auto weights = graphdl::getDefaultGraph()->getWeights();
    EXPECT_EQ(weights.size(), 1);
    EXPECT_EQ(weights.count(WEIGHTS_NAME), 1);
}

TEST_F(CoreTest, addInputWithTheSameName)
{
    graphdl::ITensorPtr input1 = graphdl::createInput("input1", {3, 224, 224});
    EXPECT_NE(input1, nullptr);
    EXPECT_THROW(
        { graphdl::ITensorPtr t = graphdl::createInput("input1", {}); },
        std::runtime_error);
}

TEST_F(CoreTest, gradients)
{
    graphdl::ITensorPtr i = graphdl::createInput("input", {1});
    graphdl::ITensorPtr w = graphdl::createWeights("weights", {1});
    graphdl::ITensorPtr output = (graphdl::constant(1., {1}) / i) * w;
    graphdl::ITensorPtr grad = graphdl::gradients(output)[w];
    graphdl::initializeGraph();

    graphdl::HostTensor iH({5.});

    auto outputs = graphdl::eval({w, grad}, {{"input", iH}});
    EXPECT_FLOAT_EQ(outputs[1][0], 1. / 5.);
}

TEST_F(CoreTest, nonScalarGradientException)
{
    graphdl::ITensorPtr i = graphdl::createInput("input", {2, 2});
    graphdl::ITensorPtr w = graphdl::createWeights("weights", {2, 2});
    graphdl::ITensorPtr o = i * w;
    graphdl::ITensorPtr grad;
    EXPECT_THROW({ grad = graphdl::gradients(o)[w]; }, std::runtime_error);
}
