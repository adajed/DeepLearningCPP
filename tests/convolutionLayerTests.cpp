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
using TestCase = std::tuple<int, int, int, Shapes, PaddingType, MemoryLocation>;

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

std::vector<PaddingType> PADDINGS = {
    // clang-format off
    PaddingType::kVALID,
    PaddingType::kSAME
    // clang-format on
};

UVec inputShape(const TestCase& testCase)
{
    UVec shape(4, 0);
    shape[0] = std::get<0>(testCase);
    shape[1] = std::get<1>(testCase);
    shape[2] = std::get<0>(std::get<3>(testCase))[0];
    shape[3] = std::get<0>(std::get<3>(testCase))[1];
    return shape;
}

UVec kernelShape(const TestCase& testCase)
{
    UVec shape(4, 0);
    shape[0] = std::get<2>(testCase);
    shape[1] = std::get<1>(testCase);
    shape[2] = std::get<1>(std::get<3>(testCase))[0];
    shape[3] = std::get<1>(std::get<3>(testCase))[1];
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
    return std::get<5>(testCase);
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

    output[1] = std::get<2>(testCase);
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

class Conv2DTest : public LayerTest,
                   public testing::WithParamInterface<TestCase>
{
  public:
    void test(const TestCase& testCase)
    {
        setup(testCase);
        LayerBuilder builder = getBuilder(testCase);
        bool correct = runTest({mInput, mKernel}, {mOutput}, builder);
        EXPECT_TRUE(correct);
    }

  private:
    float conv(RefTensor& t, RefTensor& k)
    {
        float s = 0.;
        for (Coord_iterator it = t.begin(); it != t.end(); ++it)
            s += t[it()] * k[it()];
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
        TensorShape subShape({1, std::get<1>(testCase), k[0], k[1]});

        for (Coord_iterator it = mOutput.begin(); it != mOutput.end(); ++it)
        {
            Coord cIn({it()[0], 0, it()[2] * s[0], it()[3] * s[1]});
            Coord cK({it()[1], 0, 0, 0});
            if (padding(testCase) == PaddingType::kSAME)
            {
                cIn[2] -= (k[0] - 1) / 2;
                cIn[3] -= (k[1] - 1) / 2;
            }

            RefTensor subTensor = mInput.slice(cIn, subShape);
            RefTensor subKernel = mKernel.slice(cK, subShape);
            mOutput[it()] = conv(subTensor, subKernel);
        }
    }

    LayerBuilder getBuilder(const TestCase& testCase)
    {
        return [&testCase](const HostVec& ins) {
            ITensorPtr in =
                createInput("in", inputShape(testCase), location(testCase));
            ITensorPtr ker =
                createInput("ker", kernelShape(testCase), location(testCase));

            std::string p = "SAME";
            if (padding(testCase) == PaddingType::kVALID) p = "VALID";

            ITensorPtr out = conv2D(in, ker, strides(testCase), p);
            initializeGraph();

            return HostVec({out->eval({{"in", ins[0]}, {"ker", ins[1]}})});
        };
    }

    RefTensor mInput, mKernel, mOutput, mInputGrad, mKernelGrad, mOutputGrad;
};

TEST_P(Conv2DTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, Conv2DTest,
                        Combine(ValuesIn(N), ValuesIn(C), ValuesIn(C),
                                ValuesIn(SHAPES), ValuesIn(PADDINGS),
                                ValuesIn(LOCATIONS)));
}  // namespace
