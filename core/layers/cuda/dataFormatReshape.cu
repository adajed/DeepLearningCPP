#include "layers/cuda/macros.h"
#include "layers/dataFormatReshape.h"

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
__global__ void nhwc2nchw_kernel(const float* in, float* out, int N, int C,
                                 int X, int Y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c = n % C;
    n /= C;

    if (n < N && x < X && y < Y)
        out[((n * C + c) * X + x) * Y + y] = in[((n * X + x) * Y + y) * C + c];
}

__global__ void nchw2nhwc_kernel(const float* in, float* out, int N, int X,
                                 int Y, int C)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c = n % C;
    n /= C;

    if (n < N && x < X && y < Y)
        out[((n * X + x) * Y + y) * C + c] = in[((n * C + c) * X + x) * Y + y];
}

}  // namespace

void runNhwc2NchwDevice(const float* in, float* out, int* outShape)
{
    const int TILE = 8;
    const dim3 BLOCK(TILE, TILE, TILE);
    const dim3 GRID((outShape[2] + TILE - 1) / TILE,
                    (outShape[3] + TILE - 1) / TILE,
                    (outShape[0] * outShape[1] + TILE - 1) / TILE);

    nhwc2nchw_kernel<<<GRID, BLOCK>>>(in, out, outShape[0], outShape[1],
                                      outShape[2], outShape[3]);
}

void runNchw2NhwcDevice(const float* in, float* out, int* outShape)
{
    const int TILE = 8;
    const dim3 BLOCK(TILE, TILE, TILE);
    const dim3 GRID((outShape[1] + TILE - 1) / TILE,
                    (outShape[2] + TILE - 1) / TILE,
                    (outShape[0] * outShape[3] + TILE - 1) / TILE);

    nchw2nhwc_kernel<<<GRID, BLOCK>>>(in, out, outShape[0], outShape[1],
                                      outShape[2], outShape[3]);
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
