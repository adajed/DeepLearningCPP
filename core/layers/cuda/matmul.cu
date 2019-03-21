#include "layers/matmul.h"

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
__global__ void matmulKernel(const float* x1, const float* x2, float* y, int n,
                             int m, int k)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n * k)
    {
        int xPos = id / k;
        int yPos = id % k;
        y[id] = 0;
        for (int i = 0; i < m; ++i)
            y[id] += x1[m * xPos + i] * x2[k * i + yPos];
    }
}

__global__ void matmulGradientKernel(const float* x1, const float* x2,
                                     const float* yGrad, float* x1Grad,
                                     float* x2Grad, int n, int m, int k)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n * m)
    {
        int xPos = id / m;
        int yPos = id % m;
        x1Grad[id] = 0.;
        for (int i = 0; i < k; ++i)
            x1Grad[id] += x2[k * yPos + i] * yGrad[k * xPos + i];
    }
    else
    {
        id -= n * m;
        if (id < m * k)
        {
            int xPos = id / k;
            int yPos = id % k;
            x2Grad[id] = 0.;
            for (int i = 0; i < n; ++i)
                x2Grad[id] += x1[m * i + xPos] * yGrad[k * i + yPos];
        }
    }
}

}  // namespace

void runMatmulDevice(const float* x1, const float* x2, float* y, int n, int m,
                     int k)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (n * k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    matmulKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, x2, y, n, m, k);
}

void runMatmulGradientDevice(const float* x1, const float* x2,
                             const float* yGrad, float* x1Grad, float* x2Grad,
                             int n, int m, int k)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (n * m + m * k + BLOCK_SIZE - 1) / BLOCK_SIZE;

    matmulGradientKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(x1, x2, yGrad, x1Grad,
                                                     x2Grad, n, m, k);
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
