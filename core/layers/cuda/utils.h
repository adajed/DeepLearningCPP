#ifndef GRAPHDL_CORE_LAYERS_CUDA_UTILS_H_
#define GRAPHDL_CORE_LAYERS_CUDA_UTILS_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "layers/elementwise.h"

namespace graphdl
{
namespace core
{
namespace layers
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

//! \fn fill
//! \brief Fills memory with given value.
//!
void fill(float* memory, size_t size, float value);

//! \fn runElementwiseCastFrontDevice
//! \brief Elementwise operation but will tensor casting
//!     from the first axis.
//!
void runElementwiseCastFrontDevice(const float* x1, size_t size1,
                                   const float* x2, size_t size2,
                                   float* y, Elementwise op);

}  // namespace utils
}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_CUDA_UTILS_H_
