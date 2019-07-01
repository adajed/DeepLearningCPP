#ifndef GRAPHDL_CORE_LAYERS_CUDA_REDUCE_UTILS_H_
#define GRAPHDL_CORE_LAYERS_CUDA_REDUCE_UTILS_H_

#include <cuda.h>
#include <cfloat>

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{

enum class ReduceOpCuda
{
    ///< sum of elements
    kSUM = 0,

    ///< sum of elements squared
    kSQUARED_SUM = 1,

    kMIN = 2,
    kMAX = 3,

    ///< mean of elements
    kMEAN = 4,

    ///< mean of elements squared
    kSQUARED_MEAN = 5
};

namespace
{

// represent initial value from which reduction starts
// (initial value for accumulator)
template <ReduceOpCuda op> __device__ float initialValue()
{
    return 0.;
}
template <> __device__ float initialValue<ReduceOpCuda::kMIN>()
{
    return FLT_MAX;
}
template <> __device__ float initialValue<ReduceOpCuda::kMAX>()
{
    return -FLT_MAX;
}

// represent initial transformation for elements
template <ReduceOpCuda op>
__device__ float initialReduceOp(float x)
{
    return x;
}
template <>
__device__ float initialReduceOp<ReduceOpCuda::kSQUARED_SUM>(float x)
{
    return x * x;
}
template <>
__device__ float initialReduceOp<ReduceOpCuda::kSQUARED_MEAN>(float x)
{
    return x * x;
}

// describes how to reduce elements
template <ReduceOpCuda op> __device__ float reduceOp(float f1, float f2)
{
    return f1 + f2;
}
template <> __device__ float reduceOp<ReduceOpCuda::kMIN>(float f1, float f2)
{
    return f1 < f2 ? f1 : f2;
}
template <> __device__ float reduceOp<ReduceOpCuda::kMAX>(float f1, float f2)
{
    return f1 > f2 ? f1 : f2;
}

// describes how to compute gradient
template <ReduceOpCuda op> __device__ float reduceGradientOp(float x, float y);
template <>
__device__ float reduceGradientOp<ReduceOpCuda::kSUM>(float x, float y)
{
    return 1.;
}
template <>
__device__ float reduceGradientOp<ReduceOpCuda::kMEAN>(float x, float y)
{
    return 1.;
}
template <>
__device__ float reduceGradientOp<ReduceOpCuda::kSQUARED_SUM>(float x, float y)
{
    return 2 * x;
}
template <>
__device__ float reduceGradientOp<ReduceOpCuda::kSQUARED_MEAN>(float x, float y)
{
    return 2 * x;
}
template <>
__device__ float reduceGradientOp<ReduceOpCuda::kMIN>(float x, float y)
{
    return float(x == y);
}
template <>
__device__ float reduceGradientOp<ReduceOpCuda::kMAX>(float x, float y)
{
    return float(x == y);
}


template <ReduceOpCuda op, unsigned BS>
__device__ void warpReduce(volatile float *sdata, unsigned tid)
{
    if (BS >= 64) sdata[tid] = reduceOp<op>(sdata[tid], sdata[tid + 32]);
    if (BS >= 32) sdata[tid] = reduceOp<op>(sdata[tid], sdata[tid + 16]);
    if (BS >= 16) sdata[tid] = reduceOp<op>(sdata[tid], sdata[tid + 8]);
    if (BS >= 8) sdata[tid] = reduceOp<op>(sdata[tid], sdata[tid + 4]);
    if (BS >= 4) sdata[tid] = reduceOp<op>(sdata[tid], sdata[tid + 2]);
    if (BS >= 2) sdata[tid] = reduceOp<op>(sdata[tid], sdata[tid + 1]);
}

template <ReduceOpCuda op, unsigned BS>
__global__ void reduceKernel(const float* x, float* y, size_t reduceSize, int blockPerReduce)
{
    __shared__ float sData[BS];
    x += (blockIdx.x / blockPerReduce) * reduceSize;

    int tid = threadIdx.x;
    int id = (blockIdx.x % blockPerReduce) * BS + threadIdx.x;
    float v;

    sData[tid] = initialValue<op>();
    while (id < reduceSize)
    {
        v = initialReduceOp<op>(x[id]);
        sData[tid] = reduceOp<op>(sData[tid], v);
        id += BS * blockPerReduce;
    }
    __syncthreads();

    if (BS >= 512)
    {
        if (tid < 256) sData[tid] = reduceOp<op>(sData[tid], sData[tid + 256]);
        __syncthreads();
    }
    if (BS >= 256)
    {
        if (tid < 128) sData[tid] = reduceOp<op>(sData[tid], sData[tid + 128]);
        __syncthreads();
    }
    if (BS >= 128)
    {
        if (tid < 64) sData[tid] = reduceOp<op>(sData[tid], sData[tid + 64]);
        __syncthreads();
    }

    if (tid < 32) warpReduce<op, BS>(sData, tid);

    if (op == ReduceOpCuda::kMEAN ||
        op == ReduceOpCuda::kSQUARED_MEAN)
    {
        if (tid == 0) y[blockIdx.x] = sData[0] / reduceSize;
    }
    else
    {
        if (tid == 0) y[blockIdx.x] = sData[0];
    }
}

template <ReduceOpCuda op>
__global__ void reduceGradientKernel(const float* x, const float* y,
                                     const float* yGrad, float* xGrad,
                                     size_t size, size_t reduceSize)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
    {
        if (op == ReduceOpCuda::kMEAN ||
            op == ReduceOpCuda::kSQUARED_MEAN)
        {
            xGrad[id] = yGrad[id / reduceSize] *
                        reduceGradientOp<op>(x[id], y[id / reduceSize]) / reduceSize;
        }
        else
        {
            xGrad[id] = yGrad[id / reduceSize] *
                        reduceGradientOp<op>(x[id], y[id / reduceSize]);
        }
    }
}

template <ReduceOpCuda op, unsigned BS>
__global__ void reduceFrontKernel(const float* x, float* y,
                                  size_t outSize, size_t reduceSize,
                                  int blockPerReduce)
{
    __shared__ float sData[BS];

    int tid = threadIdx.x;
    int id = ((blockIdx.x % blockPerReduce) * BS + tid) * outSize +
             (blockIdx.x / blockPerReduce);
    float v;

    sData[tid] = initialValue<op>();
    while (id < outSize * reduceSize)
    {
        v = initialReduceOp<op>(x[id]);
        sData[tid] = reduceOp<op>(sData[tid], v);
        id += BS * blockPerReduce * outSize;
    }
    __syncthreads();

    if (BS >= 512)
    {
        if (tid < 256) sData[tid] = reduceOp<op>(sData[tid], sData[tid + 256]);
        __syncthreads();
    }
    if (BS >= 256)
    {
        if (tid < 128) sData[tid] = reduceOp<op>(sData[tid], sData[tid + 128]);
        __syncthreads();
    }
    if (BS >= 128)
    {
        if (tid < 64) sData[tid] = reduceOp<op>(sData[tid], sData[tid + 64]);
        __syncthreads();
    }

    if (tid < 32) warpReduce<op, BS>(sData, tid);
    if (op == ReduceOpCuda::kMEAN ||
        op == ReduceOpCuda::kSQUARED_MEAN)
    {
        if (tid == 0) y[blockIdx.x] = sData[0] / reduceSize;
    }
    else
    {
        if (tid == 0) y[blockIdx.x] = sData[0];
    }
}

template <ReduceOpCuda op>
__global__ void reduceFrontGradientKernel(const float* x, const float* y,
                                          const float* yGrad, float* xGrad,
                                          size_t size, size_t outSize)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
    {
        if (op == ReduceOpCuda::kMEAN ||
            op == ReduceOpCuda::kSQUARED_MEAN)
        {
            int reduceSize = size / outSize;
            xGrad[id] = yGrad[id % outSize] *
                        reduceGradientOp<op>(x[id], y[id % outSize]) / reduceSize;
        }
        else
        {
            xGrad[id] = yGrad[id % outSize] *
                        reduceGradientOp<op>(x[id], y[id % outSize]);
        }
    }
}

}  // namespace

template <ReduceOpCuda op>
void reduce(const float* x, float* y, size_t outSize, size_t reduceSize)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS_PER_REDUCTION = (reduceSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int NUM_BLOCKS = NUM_BLOCKS_PER_REDUCTION * outSize;

    if (NUM_BLOCKS_PER_REDUCTION > 1)
    {
        float* temp;
        cudaMalloc(&temp, NUM_BLOCKS * sizeof(float));
        reduceKernel<op, BLOCK_SIZE><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                x, temp, reduceSize, NUM_BLOCKS_PER_REDUCTION);
        reduceKernel<op, BLOCK_SIZE><<<outSize, BLOCK_SIZE>>>(
                temp, y, NUM_BLOCKS_PER_REDUCTION, 1);
        cudaFree(temp);
    }
    else
        reduceKernel<op, BLOCK_SIZE><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                x, y, reduceSize, NUM_BLOCKS_PER_REDUCTION);
}

template <ReduceOpCuda op>
void reduceGradient(const float* x, const float* y,
                    const float* yGrad, float* xGrad,
                    size_t outSize, size_t reduceSize)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (outSize * reduceSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    reduceGradientKernel<op><<<NUM_BLOCKS, BLOCK_SIZE>>>(
            x, y, yGrad, xGrad, outSize * reduceSize, reduceSize);
}

template <ReduceOpCuda op>
void reduceFront(const float* x, float* y, size_t outSize, size_t reduceSize)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS_PER_REDUCTION = (reduceSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int NUM_BLOCKS = NUM_BLOCKS_PER_REDUCTION * outSize;

    if (NUM_BLOCKS_PER_REDUCTION > 1)
    {
        float* temp;
        cudaMalloc(&temp, NUM_BLOCKS * sizeof(float));
        reduceFrontKernel<op, BLOCK_SIZE><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                x, temp, outSize, reduceSize, NUM_BLOCKS_PER_REDUCTION);
        reduceKernel<op, BLOCK_SIZE><<<outSize, BLOCK_SIZE>>>(
                temp, y, NUM_BLOCKS_PER_REDUCTION, 1);
        cudaFree(temp);
    }
    else
        reduceFrontKernel<op, BLOCK_SIZE><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                x, y, outSize, reduceSize, NUM_BLOCKS_PER_REDUCTION);
}

template <ReduceOpCuda op>
void reduceFrontGradient(const float* x, const float* y,
                         const float* yGrad, float* xGrad,
                         size_t outSize, size_t reduceSize)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (outSize * reduceSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    reduceFrontGradientKernel<op><<<NUM_BLOCKS, BLOCK_SIZE>>>(
            x, y, yGrad, xGrad, outSize * reduceSize, outSize);
}


}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_CUDA_REDUCE_UTILS_H_
