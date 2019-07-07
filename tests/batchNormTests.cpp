#include "abstractTensor.h"
#include "batchNorm.h"
#include "graphdl_ops.h"
#include "layerTests.h"

namespace
{
#define EPS 10e-8

using TestCase = std::tuple<std::tuple<UVec, int>, MemoryLocation>;

UVec shape(const TestCase& testCase)
{
    return std::get<0>(std::get<0>(testCase));
}

int numAxes(const TestCase& testCase)
{
    return std::get<1>(std::get<0>(testCase));
}

UVec paramShape(const TestCase& testCase)
{
    UVec s = shape(testCase);
    int n = numAxes(testCase);

    UVec newShape;
    for (int i = n; i < s.size(); ++i) newShape.push_back(s[i]);

    return newShape;
}

MemoryLocation loc(const TestCase& testCase)
{
    return std::get<1>(testCase);
}

std::vector<std::tuple<UVec, int>> SHAPES = {
    // clang-format off
    {{1}, 1},
    {{10}, 1},
    {{10, 10}, 1},
    {{10, 10}, 2},
    {{2}, 1},
    {{2, 2, 2, 2}, 1},
    {{2, 2, 2, 2}, 2},
    {{2, 2, 2, 2}, 3},
    {{2, 2, 2, 2}, 4},
    {{4, 4, 4, 16}, 3},
    {{4, 2, 2, 32}, 3},
    {{4, 1, 1, 64}, 3},
    // clang-format on
};

class BatchNormTest : public LayerTest,
                      public testing::WithParamInterface<TestCase>
{
  public:
    void test(const TestCase& testCase)
    {
        setup(testCase);
        LayerBuilder builder = getBuilder(testCase);
        bool correct =
            runTest({mInput, mAlpha, mBeta}, {mOutput}, builder, 10e-4);
        EXPECT_TRUE(correct);
    }

    void testGradient(const TestCase& testCase)
    {
        setupGradient(testCase);
        LayerBuilder builder = getGradientBuilder(testCase);
        bool correct =
            runTest({mInput, mAlpha, mBeta, mOutputGrad},
                    {mInputGrad, mAlphaGrad, mBetaGrad}, builder, 10e-4);
        EXPECT_TRUE(correct);
    }

  private:
    void setup(const TestCase& testCase)
    {
        UniformGen gen(seed);
        mInput = RefTensor(shape(testCase), gen);
        mAlpha = RefTensor(paramShape(testCase), gen);
        mBeta = RefTensor(paramShape(testCase), gen);
        mOutput = RefTensor(shape(testCase));

        RefTensor mean(paramShape(testCase));
        RefTensor stddev(paramShape(testCase));

        calculateForward(testCase, mean, stddev);
    }

    void calculateForward(const TestCase& testCase, RefTensor& mean,
                          RefTensor& stddev)
    {
        UVec s = shape(testCase);
        int axes = numAxes(testCase);

        int cS = axes;
        int cE = s.size();

        int batchSize = 1;
        for (int i = 0; i < axes; ++i) batchSize *= s[i];

        Coord_iterator sBegin = shapeBegin(shape(testCase));
        Coord_iterator sEnd = shapeEnd(shape(testCase));

        for (Coord_iterator it = sBegin; it != sEnd; ++it)
            mean[it().cast(cS, cE)] += mInput[it()];
        for (int i = 0; i < mean.getCount(); ++i)
            mean.at(i) /= float(batchSize);
        for (Coord_iterator it = sBegin; it != sEnd; ++it)
        {
            Coord c = it().cast(cS, cE);
            stddev[c] += (mInput[it()] - mean[c]) * (mInput[it()] - mean[c]);
        }
        for (int i = 0; i < stddev.getCount(); ++i)
            stddev.at(i) /= float(batchSize);
        for (Coord_iterator it = sBegin; it != sEnd; ++it)
        {
            Coord c = it().cast(cS, cE);
            mOutput[it()] = mAlpha[c] * (mInput[it()] - mean[c]) /
                                std::sqrt(stddev[c] + EPS) +
                            mBeta[c];
        }
    }

    void setupGradient(const TestCase& testCase)
    {
        UniformGen gen(seed);
        mInput = RefTensor(shape(testCase), gen);
        mAlpha = RefTensor(paramShape(testCase), gen);
        mBeta = RefTensor(paramShape(testCase), gen);
        mOutputGrad = RefTensor(shape(testCase), gen);
        mOutput = RefTensor(shape(testCase));
        mInputGrad = RefTensor(shape(testCase));
        mAlphaGrad = RefTensor(paramShape(testCase));
        mBetaGrad = RefTensor(paramShape(testCase));

        RefTensor mean(paramShape(testCase));
        RefTensor stddev(paramShape(testCase));

        calculateForward(testCase, mean, stddev);

        UVec s = shape(testCase);
        int axes = numAxes(testCase);

        int cS = axes;
        int cE = s.size();

        int batchSize = 1;
        for (int i = 0; i < axes; ++i) batchSize *= s[i];

        Coord_iterator sBegin = shapeBegin(shape(testCase));
        Coord_iterator sEnd = shapeEnd(shape(testCase));

        for (Coord_iterator it = sBegin; it != sEnd; ++it)
            mBetaGrad[it().cast(cS, cE)] += mOutputGrad[it()];
        for (Coord_iterator it = sBegin; it != sEnd; ++it)
        {
            Coord c = it().cast(cS, cE);
            mAlphaGrad[c] += mOutputGrad[it()] * (mInput[it()] - mean[c]) /
                             std::sqrt(stddev[c] + EPS);
        }
        for (Coord_iterator it_x = sBegin; it_x != sEnd; ++it_x)
        {
            for (Coord_iterator it_y = sBegin; it_y != sEnd; ++it_y)
            {
                if (it_x().cast(cS, cE) == it_y().cast(cS, cE))
                {
                    Coord c = it_x().cast(cS, cE);
                    float grad;
                    if (it_x() == it_y())
                    {
                        grad = (1. - 1. / float(batchSize)) *
                               std::sqrt(stddev[c] + EPS);
                        grad -= 0.5 * (mInput[it_y()] - mean[c]) *
                                (mInput[it_y()] - mean[c]) /
                                std::sqrt(stddev[c] + EPS);
                        grad /= stddev[c] + EPS;
                        grad *= mAlpha[c];
                    }
                    else
                    {
                        grad = (-1. / float(batchSize)) *
                               std::sqrt(stddev[c] + EPS);
                        grad -= 0.5 * (mInput[it_x()] - mean[c]) *
                                (mInput[it_y()] - mean[c]) /
                                std::sqrt(stddev[c] + EPS);
                        grad /= stddev[c] + EPS;
                        grad *= mAlpha[c];
                    }
                    mInputGrad[it_x()] += mOutputGrad[it_y()] * grad;
                }
            }
        }
    }

    LayerBuilder getBuilder(const TestCase& testCase)
    {
        return [&testCase](const HostVec& ins) {
            ITensorPtr in = createInput("in", shape(testCase), loc(testCase));
            ITensorPtr alpha =
                createInput("alpha", paramShape(testCase), loc(testCase));
            ITensorPtr beta =
                createInput("beta", paramShape(testCase), loc(testCase));
            ITensorPtr out = batchNorm(in, alpha, beta, numAxes(testCase));
            initializeGraph();
            return HostVec({out->eval(
                {{"in", ins[0]}, {"alpha", ins[1]}, {"beta", ins[2]}})});
        };
    }

    LayerBuilder getGradientBuilder(const TestCase& testCase)
    {
        return [&testCase](const HostVec& ins) {
            MemoryType type = memoryLocationToType(loc(testCase));
            Graph::SPtr graph = core::getDefaultGraph();
            Tensor::SPtr in = graph->addInput(
                "in", createLayer<InputLayer>("in", shape(testCase), type));
            Tensor::SPtr alpha = graph->addInput(
                "alpha",
                createLayer<InputLayer>("alpha", paramShape(testCase), type));
            Tensor::SPtr beta = graph->addInput(
                "beta",
                createLayer<InputLayer>("beta", paramShape(testCase), type));
            Tensor::SPtr outG = graph->addInput(
                "outG", createLayer<InputLayer>("outG", shape(testCase), type));

            Layer::SPtr layer = createLayer<layers::BatchNormLayer>(
                in, alpha, beta, numAxes(testCase));
            Layer::TensorMap grads =
                layer->gradients(layer->getOutputs()[0], outG);

            std::vector<ITensorPtr> igrads = {makeAbstractTensor(grads[in]),
                                              makeAbstractTensor(grads[alpha]),
                                              makeAbstractTensor(grads[beta])};
            initializeGraph();
            return eval(igrads, {{"in", ins[0]},
                                 {"alpha", ins[1]},
                                 {"beta", ins[2]},
                                 {"outG", ins[3]}});
        };
    }

    RefTensor mInput, mAlpha, mBeta, mOutput, mInputGrad, mAlphaGrad, mBetaGrad,
        mOutputGrad;
};

TEST_P(BatchNormTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, BatchNormTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(LOCATIONS)));

class BatchNormGradientTest : public BatchNormTest
{
};
TEST_P(BatchNormGradientTest, testAPI)
{
    testGradient(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, BatchNormGradientTest,
                        Combine(ValuesIn(SHAPES),
                                ValuesIn({MemoryLocation::kHOST})));

}  // namespace
