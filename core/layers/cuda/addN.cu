#include "layers/addN.h"

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
__global__ void addNKernel(int n, size_t size, float** xs, float* y)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
    {
        y[id] = 0;
        for (int i = 0; i < n; ++i) y[id] += xs[i][id];
    }
}

__global__ void addNGradientKernel(int n, size_t size, const float* yG,
                                   float** xGs)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
    {
        for (int i = 0; i < n; ++i) xGs[i][id] = yG[id];
    }
}

}  // namespace

void runAddNDevice(int n, size_t size, float** xs, float* y)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float** xsDevice;
    cudaMalloc((void**)&xsDevice, n * sizeof(float*));
    cudaMemcpy(xsDevice, xs, n * sizeof(float*), cudaMemcpyHostToDevice);
    addNKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(n, size, xsDevice, y);
    cudaDeviceSynchronize();
    cudaFree(xsDevice);
}

void runAddNGradientDevice(int n, size_t size, float* yGrad, float** xGrads)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float** xGradsDevice;
    cudaMalloc((void**)&xGradsDevice, n * sizeof(float*));
    cudaMemcpy(xGradsDevice, xGrads, n * sizeof(float*),
               cudaMemcpyHostToDevice);
    addNGradientKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(n, size, yGrad,
                                                   xGradsDevice);
    cudaDeviceSynchronize();
    cudaFree(xGradsDevice);
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
