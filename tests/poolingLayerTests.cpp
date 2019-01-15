#include "abstractTensor.h"
#include "graphdl_ops.h"
#include "layerTests.h"
#include "pooling.h"

namespace
{
using namespace graphdl::core::layers;
using Shapes = std::tuple<Vec, Vec, Vec>;
using TestCase =
    std::tuple<int, int, Shapes, PoolingType, PaddingType, MemoryLocation>;

std::vector<Shapes> SHAPES = {
    // clang-format off
    {{2, 2}, {2, 2}, {2, 2}},
    {{4, 4}, {2, 2}, {2, 2}},
    {{28, 28}, {2, 2}, {2, 2}},
    {{4, 4}, {2, 2}, {1, 1}},
    {{8, 8}, {3, 3}, {2, 2}},
    {{4, 10}, {3, 3}, {2, 2}},
    {{8, 8}, {3, 2}, {3, 2}},
    {{8 ,8}, {3, 2}, {2, 2}},
    {{8, 8}, {2, 2}, {4, 4}},
    {{8, 8}, {3, 1}, {3, 1}},
    {{14, 14}, {14, 14}, {14, 14}} // global
    // clang-format on
};

std::vector<int> N = {1, 2, 4, 8, 16};
std::vector<int> C = {1, 3, 8, 16};

std::vector<PoolingType> POOLINGS = {
    // clang-format off
    PoolingType::kMAX,
    PoolingType::kAVERAGE
    // clang-format on
};

std::vector<PaddingType> PADDINGS = {
    // clang-format off
    PaddingType::kVALID,
    PaddingType::kSAME
    // clang-format on
};

Vec inputShape(const TestCase& testCase)
{
    Vec shape(4, 0);
    shape[0] = std::get<0>(testCase);
    shape[1] = std::get<1>(testCase);
    shape[2] = std::get<0>(std::get<2>(testCase))[0];
    shape[3] = std::get<0>(std::get<2>(testCase))[1];
    return shape;
}

Vec kernel(const TestCase& testCase)
{
    return std::get<1>(std::get<2>(testCase));
}

Vec strides(const TestCase& testCase)
{
    return std::get<2>(std::get<2>(testCase));
}

PoolingType pooling(const TestCase& testCase)
{
    return std::get<3>(testCase);
}

PaddingType padding(const TestCase& testCase)
{
    return std::get<4>(testCase);
}

MemoryLocation location(const TestCase& testCase)
{
    return std::get<5>(testCase);
}

int ceil(int x, int y)
{
    return x / y + int(x % y > 0);
}

Vec outputShape(const TestCase& testCase)
{
    Vec output = inputShape(testCase);
    Vec k = kernel(testCase);
    Vec s = strides(testCase);

    if (padding(testCase) == PaddingType::kVALID)
    {
        output[2] = ceil(output[2] - k[0] + 1, s[0]);
        output[3] = ceil(output[3] - k[1] + 1, s[1]);
    }
    else  // padding == PaddingType::kSAME
    {
        output[2] = ceil(output[2], s[0]);
        output[3] = ceil(output[3], s[1]);
    }

    return output;
}

class Pooling2DTest : public LayerTest,
                      public testing::WithParamInterface<TestCase>
{
  public:
    void test(const TestCase& testCase)
    {
        setup(testCase);
        LayerBuilder builder = getBuilder(testCase);
        bool correct = runTest({mInput}, {mOutput}, builder);
        EXPECT_TRUE(correct);
    }

    void testGradient(const TestCase& testCase)
    {
        setupGradient(testCase);
        LayerBuilder builder = getGradientBuilder(testCase);
        bool correct = runTest({mInput, mOutputGrad}, {mInputGrad}, builder);
        EXPECT_TRUE(correct);
    }

  private:
    float pool(const RefTensor& t, PoolingType pooling)
    {
        float ret;

        if (pooling == PoolingType::kMAX)
        {
            ret = t.at(0);
            for (size_t pos = 0; pos < t.getCount(); ++pos)
                ret = ret > t.at(pos) ? ret : t.at(pos);
        }
        else
        {
            ret = 0.;
            for (size_t pos = 0; pos < t.getCount(); ++pos) ret += t.at(pos);
            ret /= float(t.getCount());
        }

        return ret;
    }

    void poolGradient(Coord cIn, Coord cOut, const TestCase& testCase)
    {
        Vec k = kernel(testCase);
        TensorShape w({1, 1, k[0], k[1]});
        Coord_iterator end = shapeEnd(w);

        if (pooling(testCase) == PoolingType::kMAX)
        {
            RefTensor subTensor = mInput.slice(cIn, w);
            float m = pool(subTensor, PoolingType::kMAX);
            for (Coord_iterator it = shapeBegin(w); it != end; ++it)
                if (isInside(cIn + it(), mInput.shape()))
                    if (mInput[cIn + it()] == m)
                        mInputGrad[cIn + it()] += mOutputGrad[cOut];
        }
        else
        {
            float grad = mOutputGrad[cOut] / float(w.getCount());
            for (Coord_iterator it = shapeBegin(w); it != end; ++it)
                if (isInside(cIn + it(), mInput.shape()))
                    mInputGrad[cIn + it()] += grad;
        }
    }

    void setup(const TestCase& testCase)
    {
        UniformGen gen(seed);
        mInput = RefTensor(inputShape(testCase), gen);
        mOutput = RefTensor(outputShape(testCase));

        Vec k = kernel(testCase);
        Vec s = strides(testCase);
        TensorShape subShape({1, 1, k[0], k[1]});

        for (Coord_iterator it = mOutput.begin(); it != mOutput.end(); ++it)
        {
            Coord c(
                {it()[0], it()[1], it()[2] * int(s[0]), it()[3] * int(s[1])});
            if (padding(testCase) == PaddingType::kSAME)
            {
                c[2] -= (int(k[0]) - 1) / 2;
                c[3] -= (int(k[1]) - 1) / 2;
            }
            RefTensor subTensor = mInput.slice(c, subShape);
            mOutput[it()] = pool(subTensor, pooling(testCase));
        }
    }

    void setupGradient(const TestCase& testCase)
    {
        UniformGen gen(seed);
        mInput = RefTensor(inputShape(testCase), gen);
        mOutputGrad = RefTensor(outputShape(testCase), gen);
        mInputGrad = RefTensor(inputShape(testCase));

        Vec k = kernel(testCase);
        Vec s = strides(testCase);

        for (size_t pos = 0; pos < mInputGrad.getCount(); ++pos)
            mInputGrad.at(pos) = 0.;
        for (Coord_iterator it = mOutputGrad.begin(); it != mOutputGrad.end();
             ++it)
        {
            Coord c(
                {it()[0], it()[1], it()[2] * int(s[0]), it()[3] * int(s[1])});
            if (padding(testCase) == PaddingType::kSAME)
            {
                c[2] -= (int(k[0]) - 1) / 2;
                c[3] -= (int(k[1]) - 1) / 2;
            }
            poolGradient(c, it(), testCase);
        }
    }

    LayerBuilder getBuilder(const TestCase& testCase)
    {
        return [&testCase](const HostVec& ins) {
            ITensorPtr in =
                createInput("in", inputShape(testCase), location(testCase));
            ITensorPtr out;

            std::string p = "SAME";
            if (padding(testCase) == PaddingType::kVALID) p = "VALID";

            std::vector<int> k(
                {int(kernel(testCase)[0]), int(kernel(testCase)[1])});
            std::vector<int> s(
                {int(strides(testCase)[0]), int(strides(testCase)[1])});

            switch (pooling(testCase))
            {
            case PoolingType::kMAX: out = maxPool2D(in, k, s, p); break;
            case PoolingType::kAVERAGE: out = avgPool2D(in, k, s, p); break;
            }
            initializeGraph();

            return HostVec({out->eval({{"in", ins[0]}})});
        };
    }

    LayerBuilder getGradientBuilder(const TestCase& testCase)
    {
        return [&testCase](const HostVec& ins) {
            MemoryType type = memoryLocationToType(location(testCase));
            Tensor::SPtr in = core::getDefaultGraph()->addInput(
                "in",
                createLayer<InputLayer>("in", inputShape(testCase), type));
            Tensor::SPtr outG = core::getDefaultGraph()->addInput(
                "outG",
                createLayer<InputLayer>("outG", outputShape(testCase), type));

            std::vector<int> k(
                {int(kernel(testCase)[0]), int(kernel(testCase)[1])});
            std::vector<int> s(
                {int(strides(testCase)[0]), int(strides(testCase)[1])});

            Tensor::SPtr out =
                pooling2D(in, pooling(testCase), k, s, padding(testCase));

            Layer::SPtr layer = createLayer<Pooling2DGradientLayer>(
                in, out, outG, pooling(testCase), k, s, padding(testCase));
            ITensorPtr grad = makeAbstractTensor(layer->getOutputs()[0]);
            initializeGraph();
            return HostVec({grad->eval({{"in", ins[0]}, {"outG", ins[1]}})});
        };
    }

    RefTensor mInput, mOutput, mOutputGrad, mInputGrad;
};

TEST_P(Pooling2DTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, Pooling2DTest,
                        Combine(ValuesIn(N), ValuesIn(C), ValuesIn(SHAPES),
                                ValuesIn(POOLINGS), ValuesIn(PADDINGS),
                                ValuesIn(LOCATIONS)));

class Pooling2DGradientTest : public Pooling2DTest
{
};
TEST_P(Pooling2DGradientTest, testAPI)
{
    testGradient(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, Pooling2DGradientTest,
                        Combine(ValuesIn(N), ValuesIn(C), ValuesIn(SHAPES),
                                ValuesIn(POOLINGS), ValuesIn(PADDINGS),
                                ValuesIn(LOCATIONS)));
}  // namespace
