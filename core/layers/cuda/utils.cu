#include "utils.h"

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
namespace
{
__global__ void fillKernel(float* memory, size_t size, float value)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) memory[id] = value;
}

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

template <Elementwise elem, int b>
__global__ void elementwiseCastFrontKernel(const float* x1, size_t size,
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

size_t numBlocks(size_t numThreads, size_t blockSize)
{
    return (numThreads + blockSize - 1) / blockSize;
}

void fill(float* memory, size_t size, float value)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    fillKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(memory, size, value);
}

void runElementwiseCastFrontDevice(const float* x1, size_t size1,
                                   const float* x2, size_t size2, float* y,
                                   Elementwise op)
{
    size_t size = size1 > size2 ? size1 : size2;
    size_t reduceSize = size / (size1 < size2 ? size1 : size2);

    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (size1 > size2)
    {
        switch (op)
        {
        case Elementwise::kADD:
            elementwiseCastFrontKernel<Elementwise::kADD, 1>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size, x2, reduceSize, y);
            break;
        case Elementwise::kSUB:
            elementwiseCastFrontKernel<Elementwise::kSUB, 1>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size, x2, reduceSize, y);
            break;
        case Elementwise::kMUL:
            elementwiseCastFrontKernel<Elementwise::kMUL, 1>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size, x2, reduceSize, y);
            break;
        case Elementwise::kDIV:
            elementwiseCastFrontKernel<Elementwise::kDIV, 1>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size, x2, reduceSize, y);
            break;
        }
    }
    else
    {
        switch (op)
        {
        case Elementwise::kADD:
            elementwiseCastFrontKernel<Elementwise::kADD, 2>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size, x2, reduceSize, y);
            break;
        case Elementwise::kSUB:
            elementwiseCastFrontKernel<Elementwise::kSUB, 2>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size, x2, reduceSize, y);
            break;
        case Elementwise::kMUL:
            elementwiseCastFrontKernel<Elementwise::kMUL, 2>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size, x2, reduceSize, y);
            break;
        case Elementwise::kDIV:
            elementwiseCastFrontKernel<Elementwise::kDIV, 2>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, size, x2, reduceSize, y);
            break;
        }
    }
}

}  // namespace utils
}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
