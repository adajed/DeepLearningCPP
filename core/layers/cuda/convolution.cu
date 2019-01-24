#include "layers/convolution.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
extern "C" void runConv2DDevice(const float* x, const float* k, float* y,
                                int* shape, int* kernel, int* strides,
                                PaddingType padding)
{
}

extern "C" void runConv2DGradientDevice(const float* x, const float* k,
                                        const float* yG, float* xG, float* kG,
                                        int* shape, int* kernel, int* strides,
                                        PaddingType padding)
{
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
