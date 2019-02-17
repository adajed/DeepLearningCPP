#include "layers/elementwise.h"
#include "reduceUtils.h"

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
__global__ void elementwiseKernel(const float* x1, size_t size1,
                                  const float* x2, size_t size2,
                                  float* y)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size1 || id < size2)
        y[id] = op<elem>(x1[id % size1], x2[id % size2]);
}

template <Elementwise elem, int n>
__global__ void elementwiseGradientKernelBig(const float* x1, size_t size1,
                                             const float* x2, size_t size2,
                                             const float* yG, float* xG)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size1 || id < size2)
        xG[id] = yG[id] * opGrad<elem, n>(x1[id % size1], x2[id % size2]);
}

template <Elementwise elem, int n>
__global__ void elementwiseGradientKernelSmall(const float* x1, size_t size1,
                                               const float* x2, size_t size2,
                                               const float* yG, float* out)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t minSize = (size1 > size2) ? size2 : size1;
    size_t maxSize = (size1 > size2) ? size1 : size2;

    if (id < maxSize)
        out[(maxSize / minSize) * (id % minSize) + (id / minSize)] =
            yG[id] * opGrad<elem, n>(x1[id % size1], x2[id % size2]);
}

template <Elementwise elem>
void runElementwiseGradientKernels(const float* x1, size_t size1,
                                   const float* x2, size_t size2,
                                   const float* yG, float* x1Grad,
                                   float* x2Grad)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS =
        ((size1 > size2 ? size1 : size2) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (size1 == size2)
    {
        elementwiseGradientKernelBig<elem, 0>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, yG, x1Grad);
        elementwiseGradientKernelBig<elem, 1>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, yG, x2Grad);
        return;
    }

    float* temp;
    cudaMalloc((void**)&temp, (size1 > size2 ? size1 : size2) * sizeof(float));

    if (size1 > size2)
    {
        elementwiseGradientKernelBig<elem, 0>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, yG, x1Grad);
        elementwiseGradientKernelSmall<elem, 1>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, yG, temp);

        reduce<ReduceOpCuda::kSUM>(temp, x2Grad, size2, size1 / size2);
    }
    else
    {
        elementwiseGradientKernelSmall<elem, 0>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, yG, temp);
        elementwiseGradientKernelBig<elem, 1>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, yG, x2Grad);

        reduce<ReduceOpCuda::kSUM>(temp, x1Grad, size1, size2 / size1);
    }

    cudaFree(temp);
}

}  // namespace


void runElementwiseDevice(const float* x1, size_t size1,
                          const float* x2, size_t size2,
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
}

void runElementwiseGradientDevice(const float* x1, size_t size1,
                                  const float* x2, size_t size2,
                                  const float* yGrad, float* x1Grad,
                                  float* x2Grad, Elementwise op)
{
    switch (op)
    {
    case Elementwise::kADD:
        runElementwiseGradientKernels<Elementwise::kADD>(x1, size1, x2, size2,
                                                         yGrad, x1Grad, x2Grad);
        break;
    case Elementwise::kSUB:
        runElementwiseGradientKernels<Elementwise::kSUB>(x1, size1, x2, size2,
                                                         yGrad, x1Grad, x2Grad);
        break;
    case Elementwise::kMUL:
        runElementwiseGradientKernels<Elementwise::kMUL>(x1, size1, x2, size2,
                                                         yGrad, x1Grad, x2Grad);
        break;
    case Elementwise::kDIV:
        runElementwiseGradientKernels<Elementwise::kDIV>(x1, size1, x2, size2,
                                                         yGrad, x1Grad, x2Grad);
        break;
    }
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
