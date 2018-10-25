#include <gtest/gtest.h>
#include "dll.h"
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

TEST_F(CoreTest, evalInputOper)
{
    const int SIZE = 10;
    dll::ITensorSPtr input = dll::createInput("input", {SIZE});

    dll::initializeGraph();

    dll::HostTensor in{nullptr, SIZE};
    dll::HostTensor out{nullptr, SIZE};

    in.values = new float[SIZE];
    out.values = new float[SIZE];
    for (int i = 0; i < SIZE; ++i) in.values[i] = i;

    input->eval({{"input", in}}, out);

    for (int i = 0; i < SIZE; ++i) EXPECT_EQ(in.values[i], out.values[i]);

    delete[] in.values;
    delete[] out.values;
}

void test(int SIZE, dll::ITensorSPtr output, dll::HostTensor& in1,
          dll::HostTensor& in2, dll::HostTensor& out)
{
    in1.values = new float[SIZE];
    in2.values = new float[SIZE];
    out.values = new float[SIZE];

    for (int i = 0; i < SIZE; ++i)
    {
        in1.values[i] = dist(e2);
        in2.values[i] = dist(e2);
    }

    output->eval({{"input1", in1}, {"input2", in2}}, out);
}

TEST_F(CoreTest, add)
{
    const int SIZE = 10;
    dll::ITensorSPtr input1 = dll::createInput("input1", {SIZE});
    dll::ITensorSPtr input2 = dll::createInput("input2", {SIZE});
    dll::ITensorSPtr output = input1 + input2;
    dll::initializeGraph();

    dll::HostTensor in1{nullptr, SIZE};
    dll::HostTensor in2{nullptr, SIZE};
    dll::HostTensor out{nullptr, SIZE};

    test(SIZE, output, in1, in2, out);

    for (int i = 0; i < SIZE; ++i)
        EXPECT_EQ(out.values[i], in1.values[i] + in2.values[i]);

    delete[] in1.values;
    delete[] in2.values;
    delete[] out.values;
}

TEST_F(CoreTest, sub)
{
    const int SIZE = 10;
    dll::ITensorSPtr input1 = dll::createInput("input1", {SIZE});
    dll::ITensorSPtr input2 = dll::createInput("input2", {SIZE});
    dll::ITensorSPtr output = input1 - input2;
    dll::initializeGraph();

    dll::HostTensor in1{nullptr, SIZE};
    dll::HostTensor in2{nullptr, SIZE};
    dll::HostTensor out{nullptr, SIZE};

    test(SIZE, output, in1, in2, out);

    for (int i = 0; i < SIZE; ++i)
        EXPECT_EQ(out.values[i], in1.values[i] - in2.values[i]);

    delete[] in1.values;
    delete[] in2.values;
    delete[] out.values;
}

TEST_F(CoreTest, mul)
{
    const int SIZE = 10;
    dll::ITensorSPtr input1 = dll::createInput("input1", {SIZE});
    dll::ITensorSPtr input2 = dll::createInput("input2", {SIZE});
    dll::ITensorSPtr output = input1 * input2;
    dll::initializeGraph();

    dll::HostTensor in1{nullptr, SIZE};
    dll::HostTensor in2{nullptr, SIZE};
    dll::HostTensor out{nullptr, SIZE};

    test(SIZE, output, in1, in2, out);

    for (int i = 0; i < SIZE; ++i)
        EXPECT_EQ(out.values[i], in1.values[i] * in2.values[i]);

    delete[] in1.values;
    delete[] in2.values;
    delete[] out.values;
}

TEST_F(CoreTest, div)
{
    const int SIZE = 10;
    dll::ITensorSPtr input1 = dll::createInput("input1", {SIZE});
    dll::ITensorSPtr input2 = dll::createInput("input2", {SIZE});
    dll::ITensorSPtr output = input1 / input2;
    dll::initializeGraph();

    dll::HostTensor in1{nullptr, SIZE};
    dll::HostTensor in2{nullptr, SIZE};
    dll::HostTensor out{nullptr, SIZE};

    test(SIZE, output, in1, in2, out);

    for (int i = 0; i < SIZE; ++i)
        EXPECT_EQ(out.values[i], in1.values[i] / in2.values[i]);

    delete[] in1.values;
    delete[] in2.values;
    delete[] out.values;
}

TEST_F(CoreTest, gradients)
{
    dll::ITensorSPtr i = dll::createInput("input", {2});
    dll::ITensorSPtr w = dll::createWeights("weights", {2});
    dll::ITensorSPtr output = (dll::constant(1., {2}) / i) * w;
    dll::ITensorSPtr grad = dll::gradients(output)[w];
    dll::initializeGraph();

    dll::HostTensor iH{nullptr, 2};
    dll::HostTensor wH{nullptr, 2};
    dll::HostTensor gH{nullptr, 2};
    iH.values = new float[2];
    wH.values = new float[2];
    gH.values = new float[2];
    iH.values[0] = 5.;
    iH.values[1] = 3.;

    dll::eval({w, grad}, {{"input", iH}}, {wH, gH});
    EXPECT_FLOAT_EQ(gH.values[0], 1. / 5.);
    EXPECT_FLOAT_EQ(gH.values[1], 1. / 3.);

    delete [] iH.values;
    delete [] wH.values;
    delete [] gH.values;
}
