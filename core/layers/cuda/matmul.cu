#include "layers/matmul.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace cuda
{
template <int TILE_SIZE, bool tran1, bool tran2>
__global__ void matmulKernel(int n, int m, int k, float* X1, float* X2,
                             float* Y)
{
    __shared__ float tile_X1[TILE_SIZE * TILE_SIZE];
    __shared__ float tile_X2[TILE_SIZE * TILE_SIZE];

    int pos = TILE_SIZE * threadIdx.x + threadIdx.y;
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    float tmp = 0.;

    for (int t = 0; t < m; t += TILE_SIZE)
    {
        if (t + threadIdx.y < m)
        {
            if (tran1)
                tile_X1[pos] = X1[n * (t + threadIdx.y) + row];
            else
                tile_X1[pos] = X1[m * row + t + threadIdx.y];
        }
        else
            tile_X1[pos] = 0.;

        if (t + threadIdx.x < m)
        {
            if (tran2)
                tile_X2[pos] = X2[m * col + t + threadIdx.x];
            else
                tile_X2[pos] = X2[k * (t + threadIdx.x) + col];
        }
        else
            tile_X2[pos] = 0.;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            tmp += tile_X1[TILE_SIZE * threadIdx.x + i] *
                   tile_X2[TILE_SIZE * i + threadIdx.y];

        __syncthreads();
    }

    if (row < n && col < k) Y[k * row + col] = tmp;
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
    const int TILE_SIZE = 16;
    dim3 GRID((n + TILE_SIZE - 1) / TILE_SIZE, (k + TILE_SIZE - 1) / TILE_SIZE);
    dim3 BLOCK(TILE_SIZE, TILE_SIZE);
    matmulKernel<TILE_SIZE, false, false><<<GRID, BLOCK>>>(n, m, k, X1, X2, Y);
}

extern "C" void runMatmulGradientDevice(int n, int m, int k, float* X1,
                                        float* X2, float* Ygrad, float* X1grad,
                                        float* X2grad)
{
    const int TILE_SIZE = 16;
    dim3 BLOCK(TILE_SIZE, TILE_SIZE);
    dim3 GRID1((n + TILE_SIZE - 1) / TILE_SIZE,
               (m + TILE_SIZE - 1) / TILE_SIZE);
    dim3 GRID2((m + TILE_SIZE - 1) / TILE_SIZE,
               (k + TILE_SIZE - 1) / TILE_SIZE);

    matmulKernel<TILE_SIZE, false, true>
        <<<GRID1, BLOCK>>>(n, k, m, Ygrad, X2, X1grad);
    matmulKernel<TILE_SIZE, true, false>
        <<<GRID2, BLOCK>>>(m, n, k, X1, Ygrad, X2grad);
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
