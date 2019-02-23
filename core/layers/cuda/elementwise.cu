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
__global__ void elementwiseBackKernel(const float* x1, size_t size1,
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
void runBackGradientKernels(const float* x1, size_t size1,
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

template <Elementwise elem, int b>
__global__ void elementwiseFrontKernel(const float* x1, size_t size,
                                       const float* x2, size_t reduceSize,
                                       float* y)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
    {
        if (b == 1)
            y[id] = op<elem>(x1[id], x2[id / reduceSize]);
        else
            y[id] = op<elem>(x1[id / reduceSize], x2[id]);
    }
}

}  // namespace


void runElementwiseBackDevice(const float* x1, size_t size1,
                          const float* x2, size_t size2,
                          float* y, Elementwise op)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS =
        ((size1 > size2 ? size1 : size2) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (op)
    {
    case Elementwise::kADD:
        elementwiseBackKernel<Elementwise::kADD>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, y);
        break;
    case Elementwise::kSUB:
        elementwiseBackKernel<Elementwise::kSUB>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, y);
        break;
    case Elementwise::kMUL:
        elementwiseBackKernel<Elementwise::kMUL>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, y);
        break;
    case Elementwise::kDIV:
        elementwiseBackKernel<Elementwise::kDIV>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size1, x2, size2, y);
        break;
    }
}

void runElementwiseBackGradientDevice(const float* x1, size_t size1,
                                  const float* x2, size_t size2,
                                  const float* yGrad, float* x1Grad,
                                  float* x2Grad, Elementwise op)
{
    switch (op)
    {
    case Elementwise::kADD:
        runBackGradientKernels<Elementwise::kADD>(x1, size1, x2, size2,
                                                         yGrad, x1Grad, x2Grad);
        break;
    case Elementwise::kSUB:
        runBackGradientKernels<Elementwise::kSUB>(x1, size1, x2, size2,
                                                         yGrad, x1Grad, x2Grad);
        break;
    case Elementwise::kMUL:
        runBackGradientKernels<Elementwise::kMUL>(x1, size1, x2, size2,
                                                         yGrad, x1Grad, x2Grad);
        break;
    case Elementwise::kDIV:
        runBackGradientKernels<Elementwise::kDIV>(x1, size1, x2, size2,
                                                         yGrad, x1Grad, x2Grad);
        break;
    }
}

void runElementwiseFrontDevice(const float* x1, size_t size1,
                               const float* x2, size_t size2,
                               float* y, Elementwise op)
{
    size_t size = size1 > size2 ? size1 : size2;
    size_t reduceSize = size / (size1 < size2 ? size1 : size2);

    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

#define CASE_RUN_KERNEL(OP, NUM) \
    case Elementwise::OP: \
        elementwiseFrontKernel<Elementwise::OP, NUM> \
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size, x2, reduceSize, y); \
        break;

    if (size1 > size2)
    {
        switch (op)
        {
        CASE_RUN_KERNEL(kADD, 1)
        CASE_RUN_KERNEL(kSUB, 1)
        CASE_RUN_KERNEL(kMUL, 1)
        CASE_RUN_KERNEL(kDIV, 1)
        }
    }
    else
    {
        switch (op)
        {
        CASE_RUN_KERNEL(kADD, 2)
        CASE_RUN_KERNEL(kSUB, 2)
        CASE_RUN_KERNEL(kMUL, 2)
        CASE_RUN_KERNEL(kDIV, 2)
        }
    }

#undef CASE_RUN_KERNEL
}

void runElementwiseFrontGradientDevice(const float* x1, size_t size1,
                                      const float* x2, size_t size2,
                                      const float* yGrad, float* x1Grad,
                                      float* x2Grad, Elementwise op)
{
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
