#include <gtest/gtest.h>
#include "dll.h"
#include "graph.h"

class CoreTest : public testing::Test
{
protected:
    void SetUp() override
    {
        testing::Test::SetUp();
    }

    void TearDown() override
    {
        dll::core::GraphRegister::getGlobalGraphRegister().clear();
        testing::Test::TearDown();
    }
};

const std::string SAMPLE_NAME = "sample_name";

TEST_F(CoreTest, simple)
{
    dll::IGraph* g = dll::createGraph(SAMPLE_NAME);
    EXPECT_NE(g, nullptr);
}

TEST_F(CoreTest, graphWithGivenNameAlreadyExists)
{
    dll::IGraph* g = dll::createGraph(SAMPLE_NAME);
    EXPECT_NE(g, nullptr);
    g = dll::createGraph(SAMPLE_NAME);
    EXPECT_EQ(g, nullptr);
}

TEST_F(CoreTest, setDefaultGraph)
{
    dll::IGraph* g = dll::createGraph(SAMPLE_NAME);
    dll::setDefaultGraph(g);
    dll::IGraph* g2 = dll::getDefaultGraph();
    EXPECT_EQ(g, g2);
}

TEST_F(CoreTest, emptyInput)
{
    std::vector<dll::ITensor*> inputs = dll::getDefaultGraph()->getInputs();
    EXPECT_EQ(inputs.size(), 0);
}

TEST_F(CoreTest, addInput)
{
    dll::ITensor* input = dll::createInput("input1", {3, 224, 224});
    std::vector<dll::ITensor*> inputs = dll::getDefaultGraph()->getInputs();
    EXPECT_EQ(inputs.size(), 1);
    EXPECT_EQ(inputs[0], input);
}

TEST_F(CoreTest, addInputWithTheSameName)
{
    dll::ITensor* input1 = dll::createInput("input1", {3, 224, 224});
    EXPECT_NE(input1, nullptr);
    dll::ITensor* input2 = dll::createInput("input1", {3, 224, 224});
    EXPECT_EQ(input2, nullptr);
}

TEST_F(CoreTest, evalInputOper)
{
    const int SIZE = 10;
    dll::ITensor* input = dll::createInput("input", {SIZE});

    dll::initializeGraph();

    dll::HostTensor inT{nullptr, SIZE};
    dll::HostTensor outT{nullptr, SIZE};

    inT.values = new float[SIZE];
    outT.values = new float[SIZE];
    for (int i = 0; i < SIZE; ++i)
        inT.values[i] = i;

    input->eval({{"input", inT}}, &outT);

    for (int i = 0; i < SIZE; ++i)
        EXPECT_EQ(inT.values[i], outT.values[i]);

    delete [] inT.values;
    delete [] outT.values;
}
