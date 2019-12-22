#include "abstractTensor.h"
#include "convolution.h"
#include "graphdl_ops.h"
#include "layerTests.h"

namespace
{
using namespace graphdl::core::layers;
// (x, y), kernel, strides
using Shapes = std::tuple<UVec, Vec, Vec>;
// n, cIn, cOut, shapes, padding, location
using TestCase =
    std::tuple<int, int, int, Shapes, PaddingType, DataFormat, MemoryLocation>;

using ErrorShapes = std::tuple<UVec, UVec, Vec>;
using ErrorTestCase =
    std::tuple<ErrorShapes, PaddingType, DataFormat, MemoryLocation>;

std::vector<Shapes> SHAPES = {
    // clang-format off
    {{2, 2}, {2, 2}, {2, 2}},
    {{4, 4}, {2, 2}, {2, 2}},
    {{4, 4}, {2, 2}, {1, 1}},
    {{8, 8}, {3, 3}, {2, 2}},
    {{4, 10}, {3, 3}, {2, 2}},
    {{8, 8}, {3, 2}, {3, 2}},
    {{8 ,8}, {3, 2}, {2, 2}},
    {{8, 8}, {2, 2}, {4, 4}},
    {{8, 8}, {3, 1}, {3, 1}},
    {{14, 14}, {14, 14}, {14, 14}}, // global
    // clang-format on
};

std::vector<ErrorShapes> ERROR_SHAPES = {
    // clang-format off
    {{1}, {1, 1, 1, 1}, {2, 2}},
    {{1, 1}, {1, 1, 1, 1}, {2, 2}},
    {{1, 1, 1}, {1, 1, 1, 1}, {2, 2}},
    {{1, 1, 1, 1, 1}, {1, 1, 1, 1}, {2, 2}},
    {{2, 2, 2, 2}, {2}, {2, 2}},
    {{2, 2, 2, 2}, {2, 2}, {2, 2}},
    {{2, 2, 2, 2}, {2, 2, 2}, {2, 2}},
    {{2, 2, 2, 2}, {2, 2, 2, 2, 2}, {2, 2}},
    {{1, 2, 2, 2}, {3, 2, 2, 2}, {}},
    {{1, 2, 2, 2}, {3, 2, 2, 2}, {2, 2, 2}},
    {{1, 2, 2, 2}, {3, 2, 2, 2}, {2, 0}},
    {{1, 2, 2, 2}, {3, 2, 2, 2}, {0, 2}},
    {{1, 2, 2, 2}, {3, 2, 2, 2}, {2, -1}},
    {{1, 2, 2, 2}, {3, 2, 2, 2}, {-1, 2}},
    // clang-format on
};

std::vector<int> N = {1, 2, 4, 8};
std::vector<int> C = {1, 3, 8};

std::vector<PaddingType> PADDINGS = {
    // clang-format off
    PaddingType::kVALID,
    PaddingType::kSAME
    // clang-format on
};

std::vector<DataFormat> DATA_FORMATS = {
    // clang-format off
    DataFormat::kNHWC,
    DataFormat::kNCHW
    // clang-format on
};

DataFormat format(const TestCase& testCase)
{
    return std::get<5>(testCase);
}

UVec inputShape(const TestCase& testCase)
{
    UVec shape(4, 0);
    shape[0] = std::get<0>(testCase);

    if (format(testCase) == DataFormat::kNHWC)
    {
        shape[1] = std::get<0>(std::get<3>(testCase))[0];
        shape[2] = std::get<0>(std::get<3>(testCase))[1];
        shape[3] = std::get<1>(testCase);
    }
    else
    {
        shape[1] = std::get<1>(testCase);
        shape[2] = std::get<0>(std::get<3>(testCase))[0];
        shape[3] = std::get<0>(std::get<3>(testCase))[1];
    }

    return shape;
}

UVec kernelShape(const TestCase& testCase)
{
    UVec shape(4, 0);
    shape[0] = std::get<1>(std::get<3>(testCase))[0];
    shape[1] = std::get<1>(std::get<3>(testCase))[1];
    shape[2] = std::get<1>(testCase);
    shape[3] = std::get<2>(testCase);
    return shape;
}

Vec kernel(const TestCase& testCase)
{
    return std::get<1>(std::get<3>(testCase));
}

Vec strides(const TestCase& testCase)
{
    return std::get<2>(std::get<3>(testCase));
}

PaddingType padding(const TestCase& testCase)
{
    return std::get<4>(testCase);
}

MemoryLocation location(const TestCase& testCase)
{
    return std::get<6>(testCase);
}

int ceil(int x, int y)
{
    return x / y + int(x % y > 0);
}

UVec outputShape(const TestCase& testCase)
{
    UVec output = inputShape(testCase);
    Vec k = kernel(testCase);
    Vec s = strides(testCase);

    int x = format(testCase) == DataFormat::kNHWC ? 1 : 2;
    int y = format(testCase) == DataFormat::kNHWC ? 2 : 3;
    int c = format(testCase) == DataFormat::kNHWC ? 3 : 1;

    output[c] = std::get<2>(testCase);
    if (padding(testCase) == PaddingType::kVALID)
    {
        output[x] = ceil(output[x] - k[0] + 1, s[0]);
        output[y] = ceil(output[y] - k[1] + 1, s[1]);
    }
    else  // padding == PaddingType::kSAME
    {
        output[x] = ceil(output[x], s[0]);
        output[y] = ceil(output[y], s[1]);
    }

    return output;
}

class Conv2DTest : public LayerTest,
                   public testing::WithParamInterface<TestCase>
{
  public:
    void test(const TestCase& testCase)
    {
        setup(testCase);
        LayerBuilder builder = getBuilder(testCase);
        bool correct = runTest({mInput, mKernel}, {mOutput}, builder, 10e-2);
        EXPECT_TRUE(correct);
    }

    void testGradient(const TestCase& testCase)
    {
        setupGradient(testCase);
        LayerBuilder builder = getGradientBuilder(testCase);
        bool correct = runTest({mInput, mKernel, mOutputGrad},
                               {mInputGrad, mKernelGrad}, builder, 10e-2);
        EXPECT_TRUE(correct);
    }

  private:
    float conv(RefTensor& t, RefTensor& k, int c, const TestCase& testCase)
    {
        float s = 0.;
        for (Coord_iterator it = t.begin(); it != t.end(); ++it)
        {
            Coord cK({});
            if (format(testCase) == DataFormat::kNHWC)
                cK = Coord({it[1], it[2], it[3], c});
            else
                cK = Coord({it[2], it[3], it[1], c});

            s += t[it()] * k[cK];
        }
        return s;
    }

    void setup(const TestCase& testCase)
    {
        UniformGen gen(seed);
        mInput = RefTensor(inputShape(testCase), gen);
        mKernel = RefTensor(kernelShape(testCase), gen);
        mOutput = RefTensor(outputShape(testCase));

        Vec k = kernel(testCase);
        Vec s = strides(testCase);
        TensorShape subShape;
        if (format(testCase) == DataFormat::kNHWC)
            subShape = TensorShape({1, k[0], k[1], std::get<1>(testCase)});
        else
            subShape = TensorShape({1, std::get<1>(testCase), k[0], k[1]});

        int x = format(testCase) == DataFormat::kNHWC ? 1 : 2;
        int y = format(testCase) == DataFormat::kNHWC ? 2 : 3;
        int c = format(testCase) == DataFormat::kNHWC ? 3 : 1;

        for (Coord_iterator it = mOutput.begin(); it != mOutput.end(); ++it)
        {
            Coord cIn({});
            if (format(testCase) == DataFormat::kNHWC)
                cIn = Coord({it[0], it[1] * s[0], it[2] * s[1], 0});
            else
                cIn = Coord({it[0], 0, it[2] * s[0], it[3] * s[1]});
            if (padding(testCase) == PaddingType::kSAME)
            {
                cIn[x] -= (k[0] - 1) / 2;
                cIn[y] -= (k[1] - 1) / 2;
            }

            RefTensor subTensor = mInput.slice(cIn, subShape);
            mOutput[it()] = conv(subTensor, mKernel, it[c], testCase);
        }
    }

    void setupGradient(const TestCase& testCase)
    {
        UniformGen gen(seed);
        mInput = RefTensor(inputShape(testCase), gen);
        mKernel = RefTensor(kernelShape(testCase), gen);
        mOutputGrad = RefTensor(outputShape(testCase), gen);
        mInputGrad = RefTensor(inputShape(testCase));
        mKernelGrad = RefTensor(kernelShape(testCase));

        Vec k = kernel(testCase);
        Vec s = strides(testCase);
        TensorShape subShape;
        if (format(testCase) == DataFormat::kNHWC)
            subShape = TensorShape({1, k[0], k[1], std::get<1>(testCase)});
        else
            subShape = TensorShape({1, std::get<1>(testCase), k[0], k[1]});

        int x = format(testCase) == DataFormat::kNHWC ? 1 : 2;
        int y = format(testCase) == DataFormat::kNHWC ? 2 : 3;
        int c = format(testCase) == DataFormat::kNHWC ? 3 : 1;

        for (size_t pos = 0; pos < mInputGrad.getCount(); ++pos)
            mInputGrad.at(pos) = 0.;
        for (size_t pos = 0; pos < mKernelGrad.getCount(); ++pos)
            mKernelGrad.at(pos) = 0.;

        for (Coord_iterator i = mOutputGrad.begin(); i != mOutputGrad.end();
             ++i)
        {
            Coord cIn({});
            if (format(testCase) == DataFormat::kNHWC)
                cIn = Coord({i[0], i[1] * s[0], i[2] * s[1], 0});
            else
                cIn = Coord({i[0], 0, i[2] * s[0], i[3] * s[1]});
            if (padding(testCase) == PaddingType::kSAME)
            {
                cIn[x] -= (k[0] - 1) / 2;
                cIn[y] -= (k[1] - 1) / 2;
            }

            RefTensor t = mInput.slice(cIn, subShape);

            for (Coord_iterator j = t.begin(); j != t.end(); ++j)
            {
                Coord cK({});
                if (format(testCase) == DataFormat::kNHWC)
                    cK = Coord({j[1], j[2], j[3], i[c]});
                else
                    cK = Coord({j[2], j[3], j[1], i[c]});

                if (isInside(cIn + j(), mInputGrad.shape()))
                    mInputGrad[cIn + j()] += mOutputGrad[i()] * mKernel[cK];
                mKernelGrad[cK] += mOutputGrad[i()] * t[j()];
            }
        }
    }

    LayerBuilder getBuilder(const TestCase& testCase)
    {
        return [&testCase](const HostVec& ins) {
            ITensorPtr in =
                createInput("in", inputShape(testCase), location(testCase));
            ITensorPtr ker =
                createInput("ker", kernelShape(testCase), location(testCase));

            std::string p =
                padding(testCase) == PaddingType::kSAME ? "SAME" : "VALID";
            std::string f =
                format(testCase) == DataFormat::kNHWC ? "NHWC" : "NCHW";

            ITensorPtr out = conv2D(in, ker, strides(testCase), p, f);
            initializeGraph();

            return HostVec({out->eval({{"in", ins[0]}, {"ker", ins[1]}})});
        };
    }

    LayerBuilder getGradientBuilder(const TestCase& testCase)
    {
        return [&testCase](const HostVec& ins) {
            MemoryType type = memoryLocationToType(location(testCase));
            UVec inShape = inputShape(testCase);
            UVec kerShape = kernelShape(testCase);
            UVec outShape = outputShape(testCase);
            Tensor::SPtr in = core::getDefaultGraph()->addInput(
                "in", createLayer<InputLayer>("in", inShape, type));
            Tensor::SPtr ker = core::getDefaultGraph()->addInput(
                "ker", createLayer<InputLayer>("ker", kerShape, type));
            Tensor::SPtr outG = core::getDefaultGraph()->addInput(
                "outG", createLayer<InputLayer>("outG", outShape, type));

            Tensor::SPtr out =
                convolution2D(in, ker, strides(testCase), padding(testCase),
                              format(testCase));
            Layer::SPtr layer = createLayer<Conv2DGradientLayer>(
                in, ker, out, outG, strides(testCase), padding(testCase),
                format(testCase));

            std::vector<Tensor::SPtr> grads = layer->getOutputs();
            std::vector<ITensorPtr> igrads = {makeAbstractTensor(grads[0]),
                                              makeAbstractTensor(grads[1])};
            initializeGraph();
            return eval(igrads,
                        {{"in", ins[0]}, {"ker", ins[1]}, {"outG", ins[2]}});
        };
    }

    RefTensor mInput, mKernel, mOutput, mInputGrad, mKernelGrad, mOutputGrad;
};

class Conv2DErrorsTest : public LayerTest,
                         public testing::WithParamInterface<ErrorTestCase>
{
  public:
    void testWrongShapes(const ErrorTestCase& testCase)
    {
        UVec inShape = std::get<0>(std::get<0>(testCase));
        UVec kerShape = std::get<1>(std::get<0>(testCase));
        Vec strides = std::get<2>(std::get<0>(testCase));
        PaddingType pad = std::get<1>(testCase);
        DataFormat format = std::get<2>(testCase);
        MemoryLocation loc = std::get<3>(testCase);
        std::string p = "SAME";
        if (pad == PaddingType::kVALID) p = "VALID";
        std::string f = format == DataFormat::kNHWC ? "NHWC" : "NCHW";

        ITensorPtr in = createInput("in", inShape, loc);
        ITensorPtr ker = createInput("ker", kerShape, loc);
        ITensorPtr out;

        EXPECT_THROW({ out = conv2D(in, ker, strides, p, f); },
                     std::runtime_error);
    }
};

TEST_P(Conv2DTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, Conv2DTest,
                        Combine(ValuesIn(N), ValuesIn(C), ValuesIn(C),
                                ValuesIn(SHAPES), ValuesIn(PADDINGS),
                                ValuesIn(DATA_FORMATS), ValuesIn(LOCATIONS)));

class Conv2DGradientTest : public Conv2DTest
{
};
TEST_P(Conv2DGradientTest, testAPI)
{
    testGradient(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, Conv2DGradientTest,
                        Combine(ValuesIn(N), ValuesIn(C), ValuesIn(C),
                                ValuesIn(SHAPES), ValuesIn(PADDINGS),
                                ValuesIn(DATA_FORMATS), ValuesIn(LOCATIONS)));

TEST_P(Conv2DErrorsTest, testWrongShapes)
{
    testWrongShapes(GetParam());
};
INSTANTIATE_TEST_CASE_P(LayerTest, Conv2DErrorsTest,
                        Combine(ValuesIn(ERROR_SHAPES), ValuesIn(PADDINGS),
                                ValuesIn(DATA_FORMATS), ValuesIn(LOCATIONS)));
}  // namespace
