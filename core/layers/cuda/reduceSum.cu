#include "layers/reduceSum.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
__global__ void reduceSumKernel(size_t size, float* x, float* y)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id == 0)
    {
        y[0] = 0.;
        for (size_t pos = 0; pos < size; ++pos) y[0] += x[pos];
    }
}

__global__ void reduceSumGradientKernel(size_t size, float* yGrad, float* xGrad)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) xGrad[id] = yGrad[0];
}

extern "C" void runReduceSumDevice(std::size_t size, float* x, float* y)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    reduceSumKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y);
    cudaDeviceSynchronize();
}

extern "C" void runReduceSumGradientDevice(std::size_t size, float* yGrad,
                                           float* xGrad)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    reduceSumGradientKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(size, yGrad, xGrad);
    cudaDeviceSynchronize();
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
