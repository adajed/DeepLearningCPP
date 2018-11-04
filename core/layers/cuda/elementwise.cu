#include "layers/elementwise.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
template <Elementwise elem>
__device__ float op(float f1, float f2);
template <>
__device__ float op<Elementwise::kADD>(float f1, float f2)
{
    return f1 + f2;
}
template <>
__device__ float op<Elementwise::kSUB>(float f1, float f2)
{
    return f1 - f2;
}
template <>
__device__ float op<Elementwise::kMUL>(float f1, float f2)
{
    return f1 * f2;
}
template <>
__device__ float op<Elementwise::kDIV>(float f1, float f2)
{
    return f1 / f2;
}

template <Elementwise elem>
__device__ float opGrad1(float f1, float f2);
template <>
__device__ float opGrad1<Elementwise::kADD>(float f1, float f2)
{
    return 1.;
}
template <>
__device__ float opGrad1<Elementwise::kSUB>(float f1, float f2)
{
    return 1.;
}
template <>
__device__ float opGrad1<Elementwise::kMUL>(float f1, float f2)
{
    return f2;
}
template <>
__device__ float opGrad1<Elementwise::kDIV>(float f1, float f2)
{
    return 1. / f2;
}

template <Elementwise elem>
__device__ float opGrad2(float f1, float f2);
template <>
__device__ float opGrad2<Elementwise::kADD>(float f1, float f2)
{
    return 1.;
}
template <>
__device__ float opGrad2<Elementwise::kSUB>(float f1, float f2)
{
    return -1.;
}
template <>
__device__ float opGrad2<Elementwise::kMUL>(float f1, float f2)
{
    return f1;
}
template <>
__device__ float opGrad2<Elementwise::kDIV>(float f1, float f2)
{
    return -f1 / (f2 * f2);
}

template <Elementwise elem>
__global__ void elementwiseKernel(size_t size, float* x1, float* x2, float* y)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) y[id] = op<elem>(x1[id], x2[id]);
}

template <Elementwise elem>
__global__ void elementwiseGradientKernel(size_t size, float* x1, float* x2,
                                          float* yG, float* x1G, float* x2G)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
    {
        x1G[id] = yG[id] * opGrad1<elem>(x1[id], x2[id]);
        x2G[id] = yG[id] * opGrad2<elem>(x1[id], x2[id]);
    }
}

void runElementwiseDevice(std::size_t size, float* x1, float* x2, float* y,
                          Elementwise op)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (op)
    {
        case Elementwise::kADD:
            elementwiseKernel<Elementwise::kADD>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x1, x2, y);
            break;
        case Elementwise::kSUB:
            elementwiseKernel<Elementwise::kSUB>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x1, x2, y);
            break;
        case Elementwise::kMUL:
            elementwiseKernel<Elementwise::kMUL>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x1, x2, y);
            break;
        case Elementwise::kDIV:
            elementwiseKernel<Elementwise::kDIV>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x1, x2, y);
            break;
    }
    cudaDeviceSynchronize();
}

void runElementwiseGradientDevice(std::size_t size, float* x1, float* x2,
                                  float* yG, float* x1G, float* x2G,
                                  Elementwise op)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (op)
    {
        case Elementwise::kADD:
            elementwiseGradientKernel<Elementwise::kADD>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x1, x2, yG, x1G, x2G);
            break;
        case Elementwise::kSUB:
            elementwiseGradientKernel<Elementwise::kSUB>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x1, x2, yG, x1G, x2G);
            break;
        case Elementwise::kMUL:
            elementwiseGradientKernel<Elementwise::kMUL>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x1, x2, yG, x1G, x2G);
            break;
        case Elementwise::kDIV:
            elementwiseGradientKernel<Elementwise::kDIV>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x1, x2, yG, x1G, x2G);
            break;
    }
    cudaDeviceSynchronize();
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
