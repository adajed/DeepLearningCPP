#include "layers/batchNorm.h"
#include "layers/elementwise.h"
#include "reduceUtils.cuh"

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
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
