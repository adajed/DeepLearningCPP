#include "layers/reduceSum.h"
#include "reduceUtils.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
__global__ void reduceSumGradientKernel(const float* yGrad, float* xGrad,
                                        size_t size, size_t reduceSize)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) xGrad[id] = yGrad[id / reduceSize];
}

void runReduceSumDevice(const float* x, float* y, size_t outSize,
                        size_t reduceSize)
{
    reduce<ReduceOpCuda::kSUM>(x, y, outSize, reduceSize);
}

void runReduceSumGradientDevice(const float* yGrad, float* xGrad,
                                size_t outSize, size_t reduceSize)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (outSize * reduceSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    reduceSumGradientKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        yGrad, xGrad, outSize * reduceSize, reduceSize);
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
