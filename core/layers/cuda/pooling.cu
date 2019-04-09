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

/* template <PaddingType padding> */
/* __device__ int firstWindow(int x, int k, int s); */

/* template <> */
/* __device__ int firstWindow<PaddingType::kVALID>(int x, int k, int s) */
/* { */
/*     int c = x - k + 1; */
/*     c = c > 0 ? c : 0; */
/*     return c / s + int(c % s > 0); */
/* } */

/* template <> */
/* __device__ int firstWindow<PaddingType::kSAME>(int x, int k, int s) */
/* { */
/*     int c = x + (k - 1) / 2 - k + 1; */
/*     c = c > 0 ? c : 0; */
/*     return c / s + int(c % s > 0); */
/* } */

/* template <PaddingType> */
/* __device__ int out2in(int x, int k, int s); */

/* template <> */
/* __device__ int out2in<PaddingType::kVALID>(int x, int /1* k *1/, int s) */
/* { */
/*     return x * s; */
/* } */

/* template <> */
/* __device__ int out2in<PaddingType::kSAME>(int x, int k, int s) */
/* { */
/*     return x * s - (k - 1) / 2; */
/* } */

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

/* template <PoolingType pooling, PaddingType padding> */
/* __global__ void poolKernel(const float* in, float* out, int* info) */
/* { */
/*     size_t n = blockIdx.x * blockDim.x + threadIdx.x; */
/*     size_t outPos = n; */

/*     int y = n % Y_OUT; */
/*     n /= Y_OUT; */
/*     int x = n % X_OUT; */
/*     n /= X_OUT; */
/*     int c = n % C; */
/*     n /= C; */

/*     if (n < N) */
/*     { */
/*         size_t inPos = (n * C + c) * X_IN * Y_IN; */
/*         out[outPos] = pool2DReduceKernel<pooling>( */
/*             in + inPos, info, out2in<padding>(x, X_KER, X_STR), */
/*             out2in<padding>(y, Y_KER, Y_STR)); */
/*     } */
/* } */

/* template <PoolingType pooling, PaddingType padding> */
/* __global__ void poolGradientKernel(const float* in, const float* out, */
/*                                    const float* outG, float* inG, int* info)
 */
/* { */
/*     size_t inPos = blockIdx.x * blockDim.x + threadIdx.x; */
/*     size_t n = inPos; */

/*     int y = n % Y_IN; */
/*     n /= Y_IN; */
/*     int x = n % X_IN; */
/*     n /= X_IN; */
/*     int c = n % C; */
/*     n /= C; */

/*     if (n < N) */
/*     { */
/*         inG[inPos] = 0.; */
/*         int xOut = firstWindow<padding>(x, X_KER, X_STR); */
/*         int yOut; */

/*         while (out2in<padding>(xOut, X_KER, X_STR) <= x && xOut < X_OUT) */
/*         { */
/*             yOut = firstWindow<padding>(y, Y_KER, Y_STR); */
/*             while (out2in<padding>(yOut, Y_KER, Y_STR) <= y && yOut < Y_OUT)
 */
/*             { */
/*                 size_t outPos = n * C + c; */
/*                 outPos = outPos * X_OUT + xOut; */
/*                 outPos = outPos * Y_OUT + yOut; */

/*                 if (pooling == PoolingType::kMAX) */
/*                 { */
/*                     if (in[inPos] == out[outPos]) inG[inPos] += outG[outPos];
 */
/*                 } */
/*                 else  // pooling == PoolingType::kAVERAGE */
/*                 { */
/*                     inG[inPos] += KER_SIZE_REC * outG[outPos]; */
/*                 } */

/*                 yOut++; */
/*             } */
/*             xOut++; */
/*         } */
/*     } */
/* } */

}  // namespace

void runPool2DDevice(const float* x, float* y, const int* params, size_t size,
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
                             float* xG, const int* params, size_t size,
                             PoolingType pooling, PaddingType padding,
                             DataFormat dataFormat)
{
    /* const int TILE_SIZE = 8; */
    /* dim3 BLOCK(8, 8, 8); */
    /* const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE; */

    /* if (pooling == PoolingType::kMAX) */
    /* { */
    /*     if (padding == PaddingType::kVALID) */
    /*         poolGradientKernel<PoolingType::kMAX, PaddingType::kVALID> */
    /*             <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yG, xG, info); */
    /*     else  // padding == PaddingType::kSAME */
    /*         poolGradientKernel<PoolingType::kMAX, PaddingType::kSAME> */
    /*             <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yG, xG, info); */
    /* } */
    /* else  // pooling == PoolingType::kAVERAGE */
    /* { */
    /*     if (padding == PaddingType::kVALID) */
    /*         poolGradientKernel<PoolingType::kAVERAGE, PaddingType::kVALID> */
    /*             <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yG, xG, info); */
    /*     else  // padding == PaddingType::kSAME */
    /*         poolGradientKernel<PoolingType::kAVERAGE, PaddingType::kSAME> */
    /*             <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yG, xG, info); */
    /* } */
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
