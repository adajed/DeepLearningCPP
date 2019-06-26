#include "layers/reduce.h"
#include "reduceUtils.h"

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
__global__ void reduceSumGradientKernel(const float* yGrad, float* xGrad,
                                        size_t size, size_t reduceSize)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) xGrad[id] = yGrad[id / reduceSize];
}

__global__ void reduceSumFrontKernel(const float* x, float* y,
                                     size_t reduceSize, size_t outSize)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < outSize)
    {
        float s = 0.;
        for (int i = 0; i < reduceSize; ++i) s += x[i * outSize + id];

        y[id] = s;
    }
}

__global__ void reduceSumFrontGradientKernel(const float* yGrad, float* xGrad,
                                             size_t size, size_t outSize)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) xGrad[id] = yGrad[id % outSize];
}

}  // namespace

void runReduceBackDevice(const float* x, float* y, size_t outSize,
                         size_t reduceSize, ReduceType /*reduceType*/)
{
    reduce<ReduceOpCuda::kSUM>(x, y, outSize, reduceSize);
}

void runReduceBackGradientDevice(const float* x, const float* y,
                                 const float* yGrad, float* xGrad,
                                 size_t outSize, size_t reduceSize,
                                 ReduceType /*reduceType*/)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (outSize * reduceSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    reduceSumGradientKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        yGrad, xGrad, outSize * reduceSize, reduceSize);
}

void runReduceFrontDevice(const float* x, float* y, size_t outSize,
                          size_t reduceSize, ReduceType /*reduceType*/)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (outSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    reduceSumFrontKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, reduceSize, outSize);
}

void runReduceFrontGradientDevice(const float* x, const float* y,
                                  const float* yGrad, float* xGrad,
                                  size_t outSize, size_t reduceSize,
                                  ReduceType /*reduceType*/)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (outSize * reduceSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    reduceSumFrontGradientKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        yGrad, xGrad, outSize * reduceSize, outSize);
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
