#include "layers/reduceSum.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
template <unsigned int BS>
__device__ void warpReduce(volatile float* sdata, unsigned int tid)
{
    if (BS >= 64) sdata[tid] += sdata[tid + 32];
    if (BS >= 32) sdata[tid] += sdata[tid + 16];
    if (BS >= 16) sdata[tid] += sdata[tid + 8];
    if (BS >= 8) sdata[tid] += sdata[tid + 4];
    if (BS >= 4) sdata[tid] += sdata[tid + 2];
    if (BS >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int BS>
__global__ void reduceSumKernel(size_t size, float* x, float* y)
{
    __shared__ float sData[BS];

    int tid = threadIdx.x;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    sData[tid] = 0.;
    while (id < size)
    {
        sData[tid] += x[id];
        id += BS * gridDim.x;
    }
    __syncthreads();

    if (BS >= 512)
    {
        if (tid < 256) sData[tid] += sData[tid + 256];
        __syncthreads();
    }
    if (BS >= 256)
    {
        if (tid < 128) sData[tid] += sData[tid + 128];
        __syncthreads();
    }
    if (BS >= 128)
    {
        if (tid < 64) sData[tid] += sData[tid + 64];
        __syncthreads();
    }

    if (tid < 32) warpReduce<BS>(sData, tid);
    if (tid == 0) y[blockIdx.x] = sData[0];
}

__global__ void reduceSumGradientKernel(size_t size, float* yGrad, float* xGrad)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) xGrad[id] = yGrad[0];
}

extern "C" void runReduceSumDevice(std::size_t size, float* x, float* y)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (NUM_BLOCKS > 1)
    {
        float* temp;
        cudaMalloc((void**)&temp, NUM_BLOCKS * sizeof(float));
        reduceSumKernel<BLOCK_SIZE><<<NUM_BLOCKS, BLOCK_SIZE>>>(size, x, temp);
        reduceSumKernel<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(NUM_BLOCKS, temp, y);
        cudaDeviceSynchronize();
        cudaFree(temp);
    }
    else
        reduceSumKernel<BLOCK_SIZE><<<1, BLOCK_SIZE>>>(size, x, y);
}

extern "C" void runReduceSumGradientDevice(std::size_t size, float* yGrad,
                                           float* xGrad)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    reduceSumGradientKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(size, yGrad, xGrad);
    cudaDeviceSynchronize();
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
