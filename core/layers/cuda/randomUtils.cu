#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

namespace graphdl
{
namespace core
{
namespace cuda
{
__global__ void setupKernel(curandState* state, size_t seed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void uniformRandomKernel(curandState* state, float* memory,
                                    size_t size, float min, float max)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) memory[id] = curand_uniform(state + id) * (max - min) + min;
}

__global__ void normalRandomKernel(curandState* state, float* memory,
                                   size_t size, float mean, float stddev)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) memory[id] = stddev * curand_normal(state + id) + mean;
}

void uniformRandom(float* memory, size_t size, float min, float max,
                   size_t seed)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    curandState* states;
    cudaMalloc(&states, size * sizeof(curandState));
    setupKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(states, seed);
    uniformRandomKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(states, memory, size, min,
                                                    max);
    cudaDeviceSynchronize();
}

void normalRandom(float* memory, size_t size, float mean, float stddev,
                  size_t seed)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    curandState* states;
    cudaMalloc(&states, size * sizeof(curandState));
    setupKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(states, seed);
    normalRandomKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(states, memory, size, mean,
                                                   stddev);
    cudaDeviceSynchronize();
}

}  // namespace cuda
}  // namespace core
}  // namespace graphdl
