#ifndef GRAPHDL_CORE_LAYERS_CUDA_REDUCE_UTILS_H_
#define GRAPHDL_CORE_LAYERS_CUDA_REDUCE_UTILS_H_

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
    // y = x_1 + x_2 + x_3 + ... + x_n
    kSUM = 0,

    // y = x_1 * x_1 + x_2 * x_2 + ... + x_n * x_n
    kSQUARED_SUM = 1
};

namespace
{
//! \fn initialReduceOp
//! \brief Represents initial operation on elements
//!
template <ReduceOpCuda op>
__device__ float initialReduceOp(float x);

template <> inline
__device__ float initialReduceOp<ReduceOpCuda::kSUM>(float x)
{
    return x;
}

template <> inline
__device__ float initialReduceOp<ReduceOpCuda::kSQUARED_SUM>(float x)
{
    return x * x;
}

//! \fn reduceOp
//! \brief Describes how to reduce elements.
//!
template <ReduceOpCuda op>
__device__ float reduceOp(float f1, float f2);

template <> inline
__device__ float reduceOp<ReduceOpCuda::kSUM>(float f1, float f2)
{
    return f1 + f2;
}

template <> inline
__device__ float reduceOp<ReduceOpCuda::kSQUARED_SUM>(float f1, float f2)
{
    return f1 + f2;
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
__global__ void reduceKernel(size_t size, const float* x, float* y)
{
    __shared__ float sData[BS];

    int tid = threadIdx.x;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float v;

    sData[tid] = 0.;
    while (id < size)
    {
        v = initialReduceOp<op>(x[id]);
        sData[tid] = reduceOp<op>(sData[tid], v);
        id += BS * gridDim.x;
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
    if (tid == 0) y[blockIdx.x] = sData[0];
}

template <ReduceOpCuda op>
void singleReduce(const float* vals, float* out, size_t size)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (NUM_BLOCKS > 1)
    {
        float* temp;
        cudaMalloc((void**)&temp, NUM_BLOCKS * sizeof(float));
        reduceKernel<op, BLOCK_SIZE><<<NUM_BLOCKS, BLOCK_SIZE>>>(size, vals, temp);
        reduceKernel<op, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(NUM_BLOCKS, temp, out);
        cudaDeviceSynchronize();
        cudaFree(temp);
    }
    else
        reduceKernel<op, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(size, vals, out);
}



}  // namespace

template <ReduceOpCuda op>
void reduce(const float* x, float* y, size_t outSize, size_t reduceSize)
{
    for (size_t i = 0; i < outSize; ++i)
    {
        singleReduce<op>(x, y, reduceSize);
        x += reduceSize;
        y += 1;
    }
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_CUDA_REDUCE_UTILS_H_
