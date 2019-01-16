#include "utils.h"

namespace graphdl
{
namespace core
{
namespace cuda
{
namespace utils
{
namespace
{
__global__ void fillKernel(float* memory, size_t size, float value)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) memory[id] = value;
}

}  // namespace

void fill(float* memory, size_t size, float value)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    fillKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(memory, size, value);
    cudaDeviceSynchronize();
}

}  // namespace utils
}  // namespace cuda
}  // namespace core
}  // namespace graphdl
