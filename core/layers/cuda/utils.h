#ifndef GRAPHDL_CORE_LAYERS_CUDA_UTILS_H_
#define GRAPHDL_CORE_LAYERS_CUDA_UTILS_H_

#include <cuda.h>
#include <cuda_runtime.h>

namespace graphdl
{
namespace core
{
namespace cuda
{
namespace utils
{

template <typename T>
void copy(T* out, T* in, size_t size)
{
    cudaMemcpy((void*)out, (void*)in, size * sizeof(T), cudaMemcpyDeviceToDevice);
}

void fill(float* memory, size_t size, float value);

}  // namespace utils
}  // namespace cuda
}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_CUDA_UTILS_H_
