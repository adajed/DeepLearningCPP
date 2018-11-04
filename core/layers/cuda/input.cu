#include <cuda.h>
#include "input.h"

namespace graphdl
{
namespace core
{
namespace cuda
{
extern "C" void copyInput(float* dest, float* src, size_t N)
{
    cudaMemcpy(dest, src, N * sizeof(float), cudaMemcpyHostToDevice);
}

}  // namespace cuda
}  // namespace core
}  // namespace graphdl
