#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "weights.h"

namespace graphdl
{
namespace core
{
namespace cuda
{
__global__ void setup_kernel(curandState* state)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, id, 0, &state[id]);
}

__global__ void initWeightsKernel(curandState* state, size_t size,
                                  float* output)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) output[id] = curand_uniform(state + id) * 2. - 1.;
}

extern "C" void initWeights(float* memory, size_t size)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    curandState* states;
    cudaMalloc(&states, size * sizeof(curandState));
    setup_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(states);
    initWeightsKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(states, size, memory);
    cudaDeviceSynchronize();
}

}  // namespace cuda
}  // namespace core
}  // namespace graphdl
