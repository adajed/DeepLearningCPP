#include "layers/activation.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
template <Activation act>
__device__ float op(float f);

template <>
__device__ float op<Activation::kRELU>(float x)
{
    return x > 0. ? x : 0.;
}

template <>
__device__ float op<Activation::kSIGMOID>(float x)
{
    return 1. / (1. + exp(-x));
}

template <>
__device__ float op<Activation::kTANH>(float x)
{
    return tanhf(x);
}

template <>
__device__ float op<Activation::kSQUARE>(float x)
{
    return x * x;
}

template <>
__device__ float op<Activation::kABS>(float x)
{
    return x >= 0. ? x : -x;
}

template <>
__device__ float op<Activation::kNEG>(float x)
{
    return -x;
}

template <>
__device__ float op<Activation::kRECIPROCAL>(float x)
{
    return 1. / x;
}

template <>
__device__ float op<Activation::kLOG>(float x)
{
    return logf(x);
}

template <>
__device__ float op<Activation::kSQRT>(float x)
{
    return sqrtf(x);
}

template <Activation act>
__global__ void activationKernel(std::size_t size, float* x, float* y)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
    {
        y[id] = op<act>(x[id]);
    }
}

template <Activation act>
__device__ float opGrad(float x, float o);

template <>
__device__ float opGrad<Activation::kRELU>(float x, float o)
{
    return x >= 0. ? 1. : 0.;
}

template <>
__device__ float opGrad<Activation::kSIGMOID>(float x, float o)
{
    return o * (1. - o);
}

template <>
__device__ float opGrad<Activation::kTANH>(float x, float o)
{
    return 1. - o * o;
}

template <>
__device__ float opGrad<Activation::kSQUARE>(float x, float o)
{
    return 2. * x;
}

template <>
__device__ float opGrad<Activation::kABS>(float x, float o)
{
    return x >= 0. ? 1. : -1.;
}

template <>
__device__ float opGrad<Activation::kNEG>(float x, float o)
{
    return -1;
}

template <>
__device__ float opGrad<Activation::kRECIPROCAL>(float x, float o)
{
    return -1. * o * o;
}

template <>
__device__ float opGrad<Activation::kLOG>(float x, float o)
{
    return 1. / x;
}

template <>
__device__ float opGrad<Activation::kSQRT>(float x, float o)
{
    return 1. / (2. * o);
}

template <Activation act>
__global__ void activationGradientKernel(std::size_t size, float* x, float* y,
                                         float* yGrad, float* xGrad)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
    {
        xGrad[id] = yGrad[id] * opGrad<act>(x[id], y[id]);
    }
}

extern "C" void runActivationDevice(std::size_t size, float* x, float* y,
                                    Activation op)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (op)
    {
    case Activation::kRELU:
        activationKernel<Activation::kRELU>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y);
        break;
    case Activation::kSIGMOID:
        activationKernel<Activation::kSIGMOID>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y);
        break;
    case Activation::kTANH:
        activationKernel<Activation::kTANH>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y);
        break;
    case Activation::kSQUARE:
        activationKernel<Activation::kSQUARE>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y);
        break;
    case Activation::kABS:
        activationKernel<Activation::kABS>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y);
        break;
    case Activation::kNEG:
        activationKernel<Activation::kNEG>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y);
        break;
    case Activation::kRECIPROCAL:
        activationKernel<Activation::kRECIPROCAL>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y);
        break;
    case Activation::kLOG:
        activationKernel<Activation::kLOG>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y);
        break;
    case Activation::kSQRT:
        activationKernel<Activation::kSQRT>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y);
        break;
    }
    cudaDeviceSynchronize();
    return;
}

extern "C" void runActivationGradientDevice(std::size_t size, float* x,
                                            float* y, float* yGrad,
                                            float* xGrad, Activation op)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (op)
    {
    case Activation::kRELU:
        activationGradientKernel<Activation::kRELU>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y, yGrad, xGrad);
        break;
    case Activation::kSIGMOID:
        activationGradientKernel<Activation::kSIGMOID>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y, yGrad, xGrad);
        break;
    case Activation::kTANH:
        activationGradientKernel<Activation::kTANH>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y, yGrad, xGrad);
        break;
    case Activation::kSQUARE:
        activationGradientKernel<Activation::kSQUARE>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y, yGrad, xGrad);
        break;
    case Activation::kABS:
        activationGradientKernel<Activation::kABS>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y, yGrad, xGrad);
        break;
    case Activation::kNEG:
        activationGradientKernel<Activation::kNEG>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y, yGrad, xGrad);
        break;
    case Activation::kRECIPROCAL:
        activationGradientKernel<Activation::kRECIPROCAL>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y, yGrad, xGrad);
        break;
    case Activation::kLOG:
        activationGradientKernel<Activation::kLOG>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y, yGrad, xGrad);
        break;
    case Activation::kSQRT:
        activationGradientKernel<Activation::kSQRT>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, y, yGrad, xGrad);
        break;
    }
    cudaDeviceSynchronize();
    return;
}
}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
