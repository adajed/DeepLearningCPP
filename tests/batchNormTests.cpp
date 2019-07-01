#include "abstractTensor.h"
#include "graphdl_ops.h"
#include "layerTests.h"

namespace
{
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

  private:
    void setup(const TestCase& testCase)
    {
        UniformGen gen(seed);
        mInput = RefTensor(shape(testCase), gen);
        mAlpha = RefTensor(paramShape(testCase), gen);
        mBeta = RefTensor(paramShape(testCase), gen);
        mOutput = RefTensor(shape(testCase));

        UVec s = shape(testCase);
        int axes = numAxes(testCase);

        std::vector<int> v;
        for (int i = 0; i < s.size(); ++i)
        {
            if (i < axes)
                v.push_back(s[i]);
            else
                v.push_back(1);
        }
        TensorShape reduceShape(v);

        Coord_iterator sBegin = shapeBegin(shape(testCase));
        Coord_iterator sEnd = shapeEnd(shape(testCase));
        for (Coord_iterator it = sBegin; it != sEnd; ++it)
        {
            std::vector<int> v;
            for (int i = 0; i < s.size(); ++i)
            {
                if (i < axes)
                    v.push_back(0);
                else
                    v.push_back(it()[i]);
            }

            RefTensor slice = mInput.slice(Coord(v), reduceShape);

            float mean = 0;
            for (size_t i = 0; i < slice.getCount(); ++i) mean += slice.at(i);
            mean /= float(slice.getCount());

            float stddev = 0;
            for (size_t i = 0; i < slice.getCount(); ++i)
                stddev += (slice.at(i) - mean) * (slice.at(i) - mean);
            stddev /= float(slice.getCount());

            v = std::vector<int>();
            for (int i = 0; i < s.size() - axes; ++i)
                v.push_back(it()[i + axes]);
            Coord c(v);
            mOutput[it()] =
                mAlpha[c] * (mInput[it()] - mean) / (sqrt(stddev) + 10e-6) +
                mBeta[c];
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

    RefTensor mInput, mAlpha, mBeta, mOutput;
};

TEST_P(BatchNormTest, testAPI)
{
    test(GetParam());
}
INSTANTIATE_TEST_CASE_P(LayerTest, BatchNormTest,
                        Combine(ValuesIn(SHAPES), ValuesIn(LOCATIONS)));

}  // namespace
