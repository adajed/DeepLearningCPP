#include "layers/cuda/macros.h"
#include "layers/cuda/utils.h"
#include "layers/pooling.h"

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
// shapeParams = [inShape, outShape, kernelShape, strides]
__constant__ int shapeParams[12];

#define IN_SHAPE shapeParams
#define OUT_SHAPE (shapeParams + 4)
#define kernelX (shapeParams[8])
#define kernelY (shapeParams[9])
#define strideX (shapeParams[10])
#define strideY (shapeParams[11])

template <PaddingType padding>
__global__ void pool2D_max_nhwc_kernel(const float* in, float* out)
{
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c = n % shapeParams[7];
    n /= shapeParams[7];

    if (n < shapeParams[4] && x_out < shapeParams[5] &&
        y_out < shapeParams[6] && c < shapeParams[7])
    {
        int x_in = x_out * strideX, y_in = y_out * strideY;
        if (padding == PaddingType::kSAME)
        {
            x_in -= (kernelX - 1) / 2;
            y_in -= (kernelY - 1) / 2;
        }

        float val = 0.;
        if (x_in >= 0 && y_in >= 0)
            val = in[POS_4D(n, x_in, y_in, c, IN_SHAPE)];

        if (x_in < 0 || x_in + kernelX > shapeParams[1] || y_in < 0 ||
            y_in + kernelY > shapeParams[2])
            val = max(val, 0.);

        for (int x_iter = max(x_in, 0);
             x_iter < min(x_in + kernelX, shapeParams[1]); ++x_iter)
        {
            for (int y_iter = max(y_in, 0);
                 y_iter < min(y_in + kernelY, shapeParams[2]); ++y_iter)
            {
                val = max(val, in[POS_4D(n, x_iter, y_iter, c, IN_SHAPE)]);
            }
        }

        out[POS_4D(n, x_out, y_out, c, OUT_SHAPE)] = val;
    }
}

template <PaddingType padding>
__global__ void pool2D_avg_nhwc_kernel(const float* in, float* out)
{
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c = n % shapeParams[7];
    n /= shapeParams[7];

    if (n < shapeParams[4] && x_out < shapeParams[5] &&
        y_out < shapeParams[6] && c < shapeParams[7])
    {
        int x_in = x_out * strideX, y_in = y_out * strideY;
        if (padding == PaddingType::kSAME)
        {
            x_in -= (kernelX - 1) / 2;
            y_in -= (kernelY - 1) / 2;
        }

        float val = 0.;
        for (int x_iter = max(x_in, 0);
             x_iter < min(x_in + kernelX, shapeParams[1]); ++x_iter)
        {
            for (int y_iter = max(y_in, 0);
                 y_iter < min(y_in + kernelY, shapeParams[2]); ++y_iter)
            {
                val += in[POS_4D(n, x_iter, y_iter, c, IN_SHAPE)];
            }
        }

        out[POS_4D(n, x_out, y_out, c, OUT_SHAPE)] = val / (kernelX * kernelY);
    }
}

template <PaddingType padding>
__global__ void pool2D_max_nchw_kernel(const float* in, float* out)
{
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c = n % shapeParams[5];
    n /= shapeParams[5];

    if (n < shapeParams[4] && c < shapeParams[5] && x_out < shapeParams[6] &&
        y_out < shapeParams[7])
    {
        int x_in = x_out * strideX, y_in = y_out * strideY;
        if (padding == PaddingType::kSAME)
        {
            x_in -= (kernelX - 1) / 2;
            y_in -= (kernelY - 1) / 2;
        }

        float val = 0.;
        if (x_in >= 0 && y_in >= 0)
            val = in[POS_4D(n, c, x_in, y_in, IN_SHAPE)];

        if (x_in < 0 || x_in + kernelX > shapeParams[2] || y_in < 0 ||
            y_in + kernelY > shapeParams[3])
            val = max(val, 0.);

        for (int x_iter = max(x_in, 0);
             x_iter < min(x_in + kernelX, shapeParams[2]); ++x_iter)
        {
            for (int y_iter = max(y_in, 0);
                 y_iter < min(y_in + kernelY, shapeParams[3]); ++y_iter)
            {
                val = max(val, in[POS_4D(n, c, x_iter, y_iter, IN_SHAPE)]);
            }
        }

        out[POS_4D(n, c, x_out, y_out, OUT_SHAPE)] = val;
    }
}

template <PaddingType padding>
__global__ void pool2D_avg_nchw_kernel(const float* in, float* out)
{
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c = n % shapeParams[5];
    n /= shapeParams[5];

    if (n < shapeParams[4] && c < shapeParams[5] && x_out < shapeParams[6] &&
        y_out < shapeParams[7])
    {
        int x_in = x_out * strideX, y_in = y_out * strideY;
        if (padding == PaddingType::kSAME)
        {
            x_in -= (kernelX - 1) / 2;
            y_in -= (kernelY - 1) / 2;
        }

        float val = 0.;
        for (int x_iter = max(x_in, 0);
             x_iter < min(x_in + kernelX, shapeParams[2]); ++x_iter)
        {
            for (int y_iter = max(y_in, 0);
                 y_iter < min(y_in + kernelY, shapeParams[3]); ++y_iter)
            {
                val += in[POS_4D(n, c, x_iter, y_iter, IN_SHAPE)];
            }
        }

        out[POS_4D(n, c, x_out, y_out, OUT_SHAPE)] = val / (kernelX * kernelY);
    }
}

template <PoolingType pooling, PaddingType padding>
__global__ void pool2D_grad_nhwc_kernel(const float* in, const float* out,
                                        const float* outG, float* inG)
{
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c = n % shapeParams[7];
    n /= shapeParams[7];

    if (n < shapeParams[4] && x_out < shapeParams[5] &&
        y_out < shapeParams[6] && c < shapeParams[7])
    {
        float outVal = out[POS_4D(n, x_out, y_out, c, OUT_SHAPE)];
        float outGVal = outG[POS_4D(n, x_out, y_out, c, OUT_SHAPE)];

        int x_in = x_out * strideX, y_in = y_out * strideY;
        if (padding == PaddingType::kSAME)
        {
            x_in -= (kernelX - 1) / 2;
            y_in -= (kernelY - 1) / 2;
        }

        for (int x_iter = max(x_in, 0);
             x_iter < min(x_in + kernelX, shapeParams[1]); ++x_iter)
        {
            for (int y_iter = max(y_in, 0);
                 y_iter < min(y_in + kernelY, shapeParams[2]); ++y_iter)
            {
                if (pooling == PoolingType::kMAX)
                {
                    if (in[POS_4D(n, x_iter, y_iter, c, IN_SHAPE)] == outVal)
                        atomicAdd(&inG[POS_4D(n, x_iter, y_iter, c, IN_SHAPE)],
                                  outGVal);
                }
                if (pooling == PoolingType::kAVERAGE)
                {
                    atomicAdd(&inG[POS_4D(n, x_iter, y_iter, c, IN_SHAPE)],
                              outGVal / (kernelX * kernelY));
                }
            }
        }
    }
}

template <PoolingType pooling, PaddingType padding>
__global__ void pool2D_grad_nchw_kernel(const float* in, const float* out,
                                        const float* outG, float* inG)
{
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c = n % shapeParams[5];
    n /= shapeParams[5];

    if (n < shapeParams[4] && c < shapeParams[5] && x_out < shapeParams[6] &&
        y_out < shapeParams[7])
    {
        float outVal = out[POS_4D(n, c, x_out, y_out, OUT_SHAPE)];
        float outGVal = outG[POS_4D(n, c, x_out, y_out, OUT_SHAPE)];

        int x_in = x_out * strideX, y_in = y_out * strideY;
        if (padding == PaddingType::kSAME)
        {
            x_in -= (kernelX - 1) / 2;
            y_in -= (kernelY - 1) / 2;
        }

        for (int x_iter = max(x_in, 0);
             x_iter < min(x_in + kernelX, shapeParams[2]); ++x_iter)
        {
            for (int y_iter = max(y_in, 0);
                 y_iter < min(y_in + kernelY, shapeParams[3]); ++y_iter)
            {
                if (pooling == PoolingType::kMAX)
                {
                    if (in[POS_4D(n, c, x_iter, y_iter, IN_SHAPE)] == outVal)
                        atomicAdd(&inG[POS_4D(n, c, x_iter, y_iter, IN_SHAPE)],
                                  outGVal);
                }
                if (pooling == PoolingType::kAVERAGE)
                {
                    atomicAdd(&inG[POS_4D(n, c, x_iter, y_iter, IN_SHAPE)],
                              outGVal / (kernelX * kernelY));
                }
            }
        }
    }
}

}  // namespace

void runPool2DDevice(const float* x, float* y, const int* params,
                     PoolingType pooling, PaddingType padding,
                     DataFormat dataFormat)
{
    const int TILE = 8;
    const dim3 BLOCK(TILE, TILE, TILE);

    dim3 GRID;
    if (dataFormat == DataFormat::kNHWC)
        GRID =
            dim3((params[5] + TILE - 1) / TILE, (params[6] + TILE - 1) / TILE,
                 (params[4] * params[7] + TILE - 1) / TILE);
    else
        GRID =
            dim3((params[6] + TILE - 1) / TILE, (params[7] + TILE - 1) / TILE,
                 (params[4] * params[5] + TILE - 1) / TILE);

    cudaMemcpyToSymbol(shapeParams, params, 12 * sizeof(int));

#define LAUNCH(format, pool)                                       \
    {                                                              \
        if (padding == PaddingType::kSAME)                         \
            pool2D_##pool##_##format##_kernel<PaddingType::kSAME>  \
                <<<GRID, BLOCK>>>(x, y);                           \
        else                                                       \
            pool2D_##pool##_##format##_kernel<PaddingType::kVALID> \
                <<<GRID, BLOCK>>>(x, y);                           \
    }

    if (dataFormat == DataFormat::kNHWC)
    {
        if (pooling == PoolingType::kMAX)
            LAUNCH(nhwc, max)
        else
            LAUNCH(nhwc, avg)
    }
    else  // dataFormat == DataFormat::kNCHW
    {
        if (pooling == PoolingType::kMAX)
            LAUNCH(nchw, max)
        else
            LAUNCH(nchw, avg)
    }

#undef LAUNCH
}

void runPool2DGradientDevice(const float* x, const float* y, const float* yG,
                             float* xG, const int* params, PoolingType pooling,
                             PaddingType padding, DataFormat dataFormat)
{
    size_t size = params[0] * params[1] * params[2] * params[3];
    utils::fill(xG, size, 0.);

    const int TILE = 8;
    const dim3 BLOCK(TILE, TILE, TILE);

    dim3 GRID;
    if (dataFormat == DataFormat::kNHWC)
        GRID =
            dim3((params[5] + TILE - 1) / TILE, (params[6] + TILE - 1) / TILE,
                 (params[4] * params[7] + TILE - 1) / TILE);
    else
        GRID =
            dim3((params[6] + TILE - 1) / TILE, (params[7] + TILE - 1) / TILE,
                 (params[4] * params[5] + TILE - 1) / TILE);

    cudaMemcpyToSymbol(shapeParams, params, 12 * sizeof(int));

#define LAUNCH(format)                                                 \
    {                                                                  \
        if (pooling == PoolingType::kMAX)                              \
        {                                                              \
            if (padding == PaddingType::kSAME)                         \
                pool2D_grad##_##format##_kernel<PoolingType::kMAX,     \
                                                PaddingType::kSAME>    \
                    <<<GRID, BLOCK>>>(x, y, yG, xG);                   \
            else                                                       \
                pool2D_grad##_##format##_kernel<PoolingType::kMAX,     \
                                                PaddingType::kVALID>   \
                    <<<GRID, BLOCK>>>(x, y, yG, xG);                   \
        }                                                              \
        else                                                           \
        {                                                              \
            if (padding == PaddingType::kSAME)                         \
                pool2D_grad##_##format##_kernel<PoolingType::kAVERAGE, \
                                                PaddingType::kSAME>    \
                    <<<GRID, BLOCK>>>(x, y, yG, xG);                   \
            else                                                       \
                pool2D_grad##_##format##_kernel<PoolingType::kAVERAGE, \
                                                PaddingType::kVALID>   \
                    <<<GRID, BLOCK>>>(x, y, yG, xG);                   \
        }                                                              \
    }

    if (dataFormat == DataFormat::kNHWC)
        LAUNCH(nhwc)
    else  // dataFormat == DataFormat::kNCHW
        LAUNCH(nchw)

#undef LAUNCH
}

#undef IN_SHAPE
#undef OUT_SHAPE
#undef kernelX
#undef kernelY
#undef strideX
#undef strideY

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
