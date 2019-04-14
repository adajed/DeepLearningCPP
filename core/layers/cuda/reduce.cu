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
}

void runReduceFrontGradientDevice(const float* x, const float* y,
                                  const float* yGrad, float* xGrad,
                                  size_t outSize, size_t reduceSize,
                                  ReduceType /*reduceType*/)
{
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
