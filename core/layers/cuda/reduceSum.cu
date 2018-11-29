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
__global__ void reduceSumGradientKernel(float* yGrad, size_t size, float* xGrad)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) xGrad[id] = yGrad[0];
}

extern "C" void runReduceSumDevice(const float* x, size_t size, float* y)
{
    reduce<ReduceOpCuda::kSUM>(x, size, y);
}

extern "C" void runReduceSumGradientDevice(float* yGrad, size_t size,
                                           float* xGrad)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    reduceSumGradientKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(yGrad, size, xGrad);
    cudaDeviceSynchronize();
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
