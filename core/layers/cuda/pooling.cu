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

#define kernelX (shapeParams[8])
#define kernelY (shapeParams[9])
#define strideX (shapeParams[10])
#define strideY (shapeParams[11])

template <PaddingType padding>
__global__ void pool2D_max_nhwc_kernel(const float* in, float* out)
{
#define POS_IN(n, x, y, c) \
    ((((n)*shapeParams[1] + (x)) * shapeParams[2] + (y)) * shapeParams[3] + (c))
#define POS_OUT(n, x, y, c) \
    ((((n)*shapeParams[5] + (x)) * shapeParams[6] + (y)) * shapeParams[7] + (c))

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
        if (x_in >= 0 && y_in >= 0) val = in[POS_IN(n, x_in, y_in, c)];

        if (x_in < 0 || x_in + kernelX > shapeParams[1] || y_in < 0 ||
            y_in + kernelY > shapeParams[2])
            val = max(val, 0.);

        for (int x_iter = max(x_in, 0);
             x_iter < min(x_in + kernelX, shapeParams[1]); ++x_iter)
        {
            for (int y_iter = max(y_in, 0);
                 y_iter < min(y_in + kernelY, shapeParams[2]); ++y_iter)
            {
                val = max(val, in[POS_IN(n, x_iter, y_iter, c)]);
            }
        }

        out[POS_OUT(n, x_out, y_out, c)] = val;
    }

#undef POS_IN
#undef POS_OUT
}

template <PaddingType padding>
__global__ void pool2D_avg_nhwc_kernel(const float* in, float* out)
{
#define POS_IN(n, x, y, c) \
    ((((n)*shapeParams[1] + (x)) * shapeParams[2] + (y)) * shapeParams[3] + (c))
#define POS_OUT(n, x, y, c) \
    ((((n)*shapeParams[5] + (x)) * shapeParams[6] + (y)) * shapeParams[7] + (c))

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
                val += in[POS_IN(n, x_iter, y_iter, c)];
            }
        }

        out[POS_OUT(n, x_out, y_out, c)] = val / (kernelX * kernelY);
    }

#undef POS_IN
#undef POS_OUT
}

template <PaddingType padding>
__global__ void pool2D_max_nchw_kernel(const float* in, float* out)
{
#define POS_IN(n, c, x, y) \
    ((((n)*shapeParams[1] + (c)) * shapeParams[2] + (x)) * shapeParams[3] + (y))
#define POS_OUT(n, c, x, y) \
    ((((n)*shapeParams[5] + (c)) * shapeParams[6] + (x)) * shapeParams[7] + (y))

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
        if (x_in >= 0 && y_in >= 0) val = in[POS_IN(n, c, x_in, y_in)];

        if (x_in < 0 || x_in + kernelX > shapeParams[2] || y_in < 0 ||
            y_in + kernelY > shapeParams[3])
            val = max(val, 0.);

        for (int x_iter = max(x_in, 0);
             x_iter < min(x_in + kernelX, shapeParams[2]); ++x_iter)
        {
            for (int y_iter = max(y_in, 0);
                 y_iter < min(y_in + kernelY, shapeParams[3]); ++y_iter)
            {
                val = max(val, in[POS_IN(n, c, x_iter, y_iter)]);
            }
        }

        out[POS_OUT(n, c, x_out, y_out)] = val;
    }

#undef POS_IN
#undef POS_OUT
}

template <PaddingType padding>
__global__ void pool2D_avg_nchw_kernel(const float* in, float* out)
{
#define POS_IN(n, c, x, y) \
    ((((n)*shapeParams[1] + (c)) * shapeParams[2] + (x)) * shapeParams[3] + (y))
#define POS_OUT(n, c, x, y) \
    ((((n)*shapeParams[5] + (c)) * shapeParams[6] + (x)) * shapeParams[7] + (y))

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
                val += in[POS_IN(n, c, x_iter, y_iter)];
            }
        }

        out[POS_OUT(n, c, x_out, y_out)] = val / (kernelX * kernelY);
    }

#undef POS_IN
#undef POS_OUT
}

template <PoolingType pooling, PaddingType padding>
__global__ void pool2D_grad_nhwc_kernel(const float* in, const float* out,
                                        const float* outG, float* inG)
{
#define POS_IN(n, x, y, c) \
    ((((n)*shapeParams[1] + (x)) * shapeParams[2] + (y)) * shapeParams[3] + (c))
#define POS_OUT(n, x, y, c) \
    ((((n)*shapeParams[5] + (x)) * shapeParams[6] + (y)) * shapeParams[7] + (c))

    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c = n % shapeParams[7];
    n /= shapeParams[7];

    if (n < shapeParams[4] && x_out < shapeParams[5] &&
        y_out < shapeParams[6] && c < shapeParams[7])
    {
        float outVal = out[POS_OUT(n, x_out, y_out, c)];
        float outGVal = outG[POS_OUT(n, x_out, y_out, c)];

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
                    if (in[POS_IN(n, x_iter, y_iter, c)] == outVal)
                        atomicAdd(&inG[POS_IN(n, x_iter, y_iter, c)], outGVal);
                }
                if (pooling == PoolingType::kAVERAGE)
                {
                    atomicAdd(&inG[POS_IN(n, x_iter, y_iter, c)],
                              outGVal / (kernelX * kernelY));
                }
            }
        }
    }

#undef POS_IN
#undef POS_OUT
}

template <PoolingType pooling, PaddingType padding>
__global__ void pool2D_grad_nchw_kernel(const float* in, const float* out,
                                        const float* outG, float* inG)
{
#define POS_IN(n, c, x, y) \
    ((((n)*shapeParams[1] + (c)) * shapeParams[2] + (x)) * shapeParams[3] + (y))
#define POS_OUT(n, c, x, y) \
    ((((n)*shapeParams[5] + (c)) * shapeParams[6] + (x)) * shapeParams[7] + (y))

    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c = n % shapeParams[5];
    n /= shapeParams[5];

    if (n < shapeParams[4] && c < shapeParams[5] && x_out < shapeParams[6] &&
        y_out < shapeParams[7])
    {
        float outVal = out[POS_OUT(n, c, x_out, y_out)];
        float outGVal = outG[POS_OUT(n, c, x_out, y_out)];

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
                    if (in[POS_IN(n, c, x_iter, y_iter)] == outVal)
                        atomicAdd(&inG[POS_IN(n, c, x_iter, y_iter)], outGVal);
                }
                if (pooling == PoolingType::kAVERAGE)
                {
                    atomicAdd(&inG[POS_IN(n, c, x_iter, y_iter)],
                              outGVal / (kernelX * kernelY));
                }
            }
        }
    }

#undef POS_IN
#undef POS_OUT
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

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
