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
__global__ void setup_kernel(curandState* state) {}

__global__ void initWeightsKernel(curandState* cuState, size_t N, float* output)
{
}

extern "C" void initWeights(float* memory, size_t N) {}

}  // namespace cuda
}  // namespace core
}  // namespace graphdl
