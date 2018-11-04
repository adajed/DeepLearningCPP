#include "layers/constant.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
__global__ void constantKernel(size_t size, float* x, float val)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) x[id] = val;
}

void fillWithValue(std::size_t size, float* x, float val)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    constantKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, val);
    cudaDeviceSynchronize();
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
