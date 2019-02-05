#include "layers/activation.h"
#include "layers/reduceSum.h"
#include "layers/softmax.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
namespace
{
__global__ void softmaxKernel(float* y, float* w, size_t size,
                              size_t reduceSize)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) y[id] /= w[id / reduceSize];
}

__global__ void softmaxGradientKernel(const float* y, const float* yGrad,
                                      float* xGrad, size_t outSize,
                                      size_t reduceSize)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < outSize * reduceSize)
    {
        xGrad[id] = yGrad[id];
        int pos = (id / reduceSize) * reduceSize;
        for (int i = 0; i < reduceSize; ++i)
        {
            xGrad[id] -= yGrad[pos] * y[pos];
            pos++;
        }
        xGrad[id] *= y[id];
    }
}

}  // namespace

void runSoftmaxDevice(const float* x, float* w, float* y, size_t outSize,
                      size_t reduceSize)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (outSize * reduceSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    runActivationDevice(x, y, outSize * reduceSize, Activation::kEXP);
    runReduceSumDevice(y, w, outSize, reduceSize);
    softmaxKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(y, w, outSize * reduceSize,
                                              reduceSize);
}

void runSoftmaxGradientDevice(const float* x, const float* y,
                              const float* yGrad, float* xGrad, size_t outSize,
                              size_t reduceSize)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (outSize * reduceSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    softmaxGradientKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(y, yGrad, xGrad, outSize,
                                                      reduceSize);
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
