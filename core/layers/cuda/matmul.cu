#include "layers/matmul.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
__global__ void matmulKernel(int n, int m, int k, float* X1, float* X2,
                             float* Y)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n * k)
    {
        int x = id / k;
        int y = id % k;
        Y[id] = 0;
        for (int i = 0; i < m; ++i) Y[id] += X1[m * x + i] * X2[k * i + y];
    }
}

__global__ void matmulGradientKernel(int n, int m, int k, float* X1, float* X2,
                                     float* Ygrad, float* X1grad, float* X2grad)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n * m)
    {
        int x = id / m;
        int y = id % m;
        X1grad[id] = 0.;
        for (int i = 0; i < k; ++i)
            X1grad[id] += X2[k * y + i] * Ygrad[k * x + i];
    }
    else
    {
        id -= n * m;
        if (id < m * k)
        {
            int x = id / k;
            int y = id % k;
            X2grad[id] = 0.;
            for (int i = 0; i < n; ++i)
                X2grad[id] += X1[m * i + x] * Ygrad[k * i + y];
        }
    }
}

extern "C" void runMatmulDevice(int n, int m, int k, float* X1, float* X2,
                             float* Y)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (n * k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    matmulKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(n, m, k, X1, X2, Y);
    cudaDeviceSynchronize();
}

extern "C" void runMatmulGradientDevice(int n, int m, int k, float* X1, float* X2,
                                     float* Ygrad, float* X1grad, float* X2grad)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (n * m + m * k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    matmulGradientKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(n, m, k, X1, X2, Ygrad,
                                                     X1grad, X2grad);
    cudaDeviceSynchronize();
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
