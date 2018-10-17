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
    dll::IGraphUPtr g = dll::createGraph(SAMPLE_NAME);
    EXPECT_NE(g.get(), nullptr);
}

TEST_F(CoreTest, graphWithGivenNameAlreadyExists)
{
    dll::IGraphUPtr g = dll::createGraph(SAMPLE_NAME);
    EXPECT_NE(g.get(), nullptr);
    g = dll::createGraph(SAMPLE_NAME);
    EXPECT_EQ(g.get(), nullptr);
}

TEST_F(CoreTest, setDefaultGraph)
{
    dll::IGraphUPtr g = dll::createGraph(SAMPLE_NAME);
    dll::setDefaultGraph(g);
    dll::IGraphUPtr g2 = dll::getDefaultGraph();
    EXPECT_EQ(g, g2);
}
