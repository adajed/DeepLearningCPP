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
template <int TILE_SIZE, bool tran1, bool tran2>
__global__ void matmulKernel(int n, int m, int k, const float* X1, const float* X2,
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

}  // namespace

void runMatmulDevice(const float* x1, const float* x2, float* y, int n, int m,
                     int k)
{
    const int TILE_SIZE = 16;
    dim3 GRID((n + TILE_SIZE - 1) / TILE_SIZE, (k + TILE_SIZE - 1) / TILE_SIZE);
    dim3 BLOCK(TILE_SIZE, TILE_SIZE);
    matmulKernel<TILE_SIZE, false, false><<<GRID, BLOCK>>>(n, m, k, x1, x2, y);
}

void runMatmulGradientDevice(const float* x1, const float* x2,
                             const float* yGrad, float* x1Grad, float* x2Grad,
                             int n, int m, int k)
{
    const int TILE_SIZE = 16;
    dim3 BLOCK(TILE_SIZE, TILE_SIZE);
    dim3 GRID1((n + TILE_SIZE - 1) / TILE_SIZE,
               (m + TILE_SIZE - 1) / TILE_SIZE);
    dim3 GRID2((m + TILE_SIZE - 1) / TILE_SIZE,
               (k + TILE_SIZE - 1) / TILE_SIZE);

    matmulKernel<TILE_SIZE, false, true>
        <<<GRID1, BLOCK>>>(n, k, m, yGrad, x2, x1Grad);
    matmulKernel<TILE_SIZE, true, false>
        <<<GRID2, BLOCK>>>(m, n, k, x1, yGrad, x2Grad);
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
