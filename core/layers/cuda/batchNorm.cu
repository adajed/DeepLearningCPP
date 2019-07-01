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

template <unsigned BS>
__device__ void warpReduce(volatile float* sdata, unsigned tid)
{
    if (BS >= 64) sdata[tid] += sdata[tid + 32];
    if (BS >= 32) sdata[tid] += sdata[tid + 16];
    if (BS >= 16) sdata[tid] += sdata[tid + 8];
    if (BS >= 8) sdata[tid] += sdata[tid + 4];
    if (BS >= 4) sdata[tid] += sdata[tid + 2];
    if (BS >= 2) sdata[tid] += sdata[tid + 1];
}

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
                                size_t batchNorm)
{
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
