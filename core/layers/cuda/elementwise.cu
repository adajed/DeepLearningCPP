#include "layers/elementwise.h"
#include "utils.h"

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

template <Elementwise elem, int n>
__device__ float opGrad(float f1, float f2);

template <>
__device__ float opGrad<Elementwise::kADD, 0>(float f1, float f2)
{
    return 1.;
}
template <>
__device__ float opGrad<Elementwise::kADD, 1>(float f1, float f2)
{
    return 1.;
}

template <>
__device__ float opGrad<Elementwise::kSUB, 0>(float f1, float f2)
{
    return 1.;
}
template <>
__device__ float opGrad<Elementwise::kSUB, 1>(float f1, float f2)
{
    return -1.;
}
template <>
__device__ float opGrad<Elementwise::kMUL, 0>(float f1, float f2)
{
    return f2;
}
template <>
__device__ float opGrad<Elementwise::kMUL, 1>(float f1, float f2)
{
    return f1;
}
template <>
__device__ float opGrad<Elementwise::kDIV, 0>(float f1, float f2)
{
    return 1. / f2;
}
template <>
__device__ float opGrad<Elementwise::kDIV, 1>(float f1, float f2)
{
    return -f1 / (f2 * f2);
}

template <Elementwise elem>
__global__ void elementwiseKernel(float* x1, size_t size1, float* x2,
                                  size_t size2, float* y)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size1 || id < size2)
        y[id] = op<elem>(x1[id % size1], x2[id % size2]);
}

template <Elementwise elem, int n>
__global__ void elementwiseGradientKernelBig(float* x1, size_t size1, float* x2,
                                             size_t size2, float* yG, float* xG)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size1 || id < size2)
        xG[id] = yG[id] * opGrad<elem, n>(x1[id % size1], x2[id % size2]);
}

template <Elementwise elem, int n>
__global__ void elementwiseGradientKernelSmall(float* x1, size_t size1,
                                               float* x2, size_t size2,
                                               float* yG, float* out)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t minSize = (size1 > size2) ? size2 : size1;
    size_t maxSize = (size1 > size2) ? size1 : size2;

    if (id < maxSize)
        out[(maxSize / minSize) * (id % minSize) + (id / minSize)] =
            yG[id] * opGrad<elem, n>(x1[id % size1], x2[id % size2]);
}

void runElementwiseDevice(float* x1, size_t size1, float* x2, size_t size2,
                          float* y, Elementwise op)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS =
        ((size1 > size2 ? size1 : size2) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (op)
    {
    case Elementwise::kADD:
        elementwiseKernel<Elementwise::kADD>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, y);
        break;
    case Elementwise::kSUB:
        elementwiseKernel<Elementwise::kSUB>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, y);
        break;
    case Elementwise::kMUL:
        elementwiseKernel<Elementwise::kMUL>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, y);
        break;
    case Elementwise::kDIV:
        elementwiseKernel<Elementwise::kDIV>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, y);
        break;
    }
    cudaDeviceSynchronize();
}

template <Elementwise elem>
void runElementwiseGradientKernels(float* x1, size_t size1, float* x2,
                                   size_t size2, float* yG, float* x1G,
                                   float* x2G)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS =
        ((size1 > size2 ? size1 : size2) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (size1 == size2)
    {
        elementwiseGradientKernelBig<elem, 0>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, yG, x1G);
        elementwiseGradientKernelBig<elem, 1>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, yG, x2G);
        return;
    }

    float* temp;
    cudaMalloc((void**)&temp, (size1 > size2 ? size1 : size2) * sizeof(float));

    if (size1 > size2)
    {
        elementwiseGradientKernelBig<elem, 0>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, yG, x1G);
        elementwiseGradientKernelSmall<elem, 1>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, yG, temp);

        for (int i = 0; i < size2; ++i)
            reduce<ReduceOpCuda::kSUM>(temp + i * (size1 / size2),
                                       size1 / size2, x2G + i);
    }
    else
    {
        elementwiseGradientKernelSmall<elem, 0>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, yG, temp);
        elementwiseGradientKernelBig<elem, 1>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, yG, x2G);

        for (int i = 0; i < size1; ++i)
            reduce<ReduceOpCuda::kSUM>(temp + i * (size2 / size1),
                                       size2 / size1, x1G + i);
    }

    cudaFree(temp);
}

void runElementwiseGradientDevice(float* x1, size_t size1, float* x2,
                                  size_t size2, float* yG, float* x1G,
                                  float* x2G, Elementwise op)
{
    switch (op)
    {
    case Elementwise::kADD:
        runElementwiseGradientKernels<Elementwise::kADD>(x1, size1, x2, size2,
                                                         yG, x1G, x2G);
        break;
    case Elementwise::kSUB:
        runElementwiseGradientKernels<Elementwise::kSUB>(x1, size1, x2, size2,
                                                         yG, x1G, x2G);
        break;
    case Elementwise::kMUL:
        runElementwiseGradientKernels<Elementwise::kMUL>(x1, size1, x2, size2,
                                                         yG, x1G, x2G);
        break;
    case Elementwise::kDIV:
        runElementwiseGradientKernels<Elementwise::kDIV>(x1, size1, x2, size2,
                                                         yG, x1G, x2G);
        break;
    }
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
