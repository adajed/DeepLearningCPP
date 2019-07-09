#include "layers/batchNorm.h"
#include "layers/elementwise.h"
#include "reduceUtils.cuh"
#include "utils.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
namespace
{
#define EPS 10e-8

__global__ void normalizeKernel(const float* alpha, const float* beta,
                                const float* stddev, float* y,
                                size_t featureSize, size_t size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
    {
        int id2 = id % featureSize;
        y[id] = alpha[id2] * y[id] / sqrt(stddev[id2] + EPS) + beta[id2];
    }
}

__global__ void alphaGradKernel(const float* betaGrad, const float* beta,
                                const float* alpha, float* alphaGrad,
                                size_t featureSize)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < featureSize)
        alphaGrad[id] = (alphaGrad[id] - betaGrad[id] * beta[id]) / alpha[id];
}

__global__ void xGradKernel(const float* x, const float* alpha, const float* y,
                            const float* yGrad, const float* mean,
                            const float* stddev, const float* betaGrad,
                            float* xGrad, size_t featureSize, size_t batchSize)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < featureSize * batchSize)
    {
        int c = id / batchSize;
        int pos = (id - c * batchSize) * featureSize + c;

        float val = -betaGrad[c] * mean[c];
        for (int i = c; i < featureSize * batchSize; i += featureSize)
            val += yGrad[i] * x[i];

        float out = yGrad[pos] - betaGrad[c] / float(batchSize);
        out -= 0.5 * (x[pos] - mean[c]) * val / (stddev[c] + EPS);
        out /= sqrt(stddev[c] + EPS);
        out *= alpha[c];
        xGrad[pos] = out;
    }
}

}  // namespace

void runBatchNormDevice(const float* x, const float* alpha, const float* beta,
                        float* y, float* mean, float* stddev, size_t size,
                        size_t batchSize)
{
    int BLOCK_SIZE = 256;
    int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t featureSize = size / batchSize;

    reduceFront<ReduceOpCuda::kMEAN>(x, mean, featureSize, batchSize);
    runElementwiseBackDevice(x, size, mean, featureSize, y, Elementwise::kSUB);
    reduceFront<ReduceOpCuda::kSQUARED_MEAN>(y, stddev, featureSize, batchSize);

    normalizeKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(alpha, beta, stddev, y,
                                                featureSize, size);
}

void runBatchNormGradientDevice(const float* x, const float* alpha,
                                const float* beta, const float* y,
                                const float* yGrad, const float* mean,
                                const float* stddev, float* xGrad,
                                float* alphaGrad, float* betaGrad, size_t size,
                                size_t batchSize)
{
    size_t featureSize = size / batchSize;

    reduceFront<ReduceOpCuda::kSUM>(yGrad, betaGrad, featureSize, batchSize);
    reduceBinFront<ReduceBinOpCuda::kDOT_PRODUCT>(yGrad, y, alphaGrad,
                                                  featureSize, batchSize);

    int BLOCK_SIZE = 256;
    int NUM_BLOCKS = utils::numBlocks(featureSize, BLOCK_SIZE);
    alphaGradKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(betaGrad, beta, alpha,
                                                alphaGrad, featureSize);

    NUM_BLOCKS = utils::numBlocks(size, BLOCK_SIZE);
    xGradKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(x, alpha, y, yGrad, mean, stddev,
                                            betaGrad, xGrad, featureSize,
                                            batchSize);
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
