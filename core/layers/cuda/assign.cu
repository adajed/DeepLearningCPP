#include "layers/assign.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
void assignDevice(float* dest, float* src, std::size_t size)
{
    cudaMemcpy(dest, src, size * sizeof(float), cudaMemcpyDeviceToDevice);
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
