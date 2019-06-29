#include "layers/convolution.h"
#include "layers/cuda/macros.h"
#include "layers/cuda/utils.h"

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
// params = [inShape, outShape, kerShape, strides]
__constant__ int shapeParams[14];

#define IN_SHAPE shapeParams
#define OUT_SHAPE (shapeParams + 4)
#define KER_SHAPE (shapeParams + 8)
#define strideX (shapeParams[12])
#define strideY (shapeParams[13])

template <PaddingType padding>
__global__ void conv2D_nhwc_kernel(const float* in, const float* ker,
                                   float* out)
{
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c_out = n % OUT_SHAPE[3];
    n /= OUT_SHAPE[3];

    if (n < OUT_SHAPE[0] && x_out < OUT_SHAPE[1] && y_out < OUT_SHAPE[2])
    {
        float val = 0;

        int x_in = x_out * strideX, y_in = y_out * strideY;
        if (padding == PaddingType::kSAME)
        {
            x_in -= (KER_SHAPE[0] - 1) / 2;
            y_in -= (KER_SHAPE[1] - 1) / 2;
        }

        for (int dx = x_in < 0 ? -x_in : 0; dx < KER_SHAPE[0]; ++dx)
        {
            if (x_in + dx >= IN_SHAPE[1]) break;
            for (int dy = y_in < 0 ? -y_in : 0; dy < KER_SHAPE[1]; ++dy)
            {
                if (y_in + dy >= IN_SHAPE[2]) break;
                for (int c_in = 0; c_in < KER_SHAPE[2]; ++c_in)
                    val += in[POS_4D(n, x_in + dx, y_in + dy, c_in, IN_SHAPE)] *
                           ker[POS_4D(dx, dy, c_in, c_out, KER_SHAPE)];
            }
        }

        out[POS_4D(n, x_out, y_out, c_out, OUT_SHAPE)] = val;
    }
}

template <PaddingType padding>
__global__ void conv2D_nchw_kernel(const float* in, const float* ker,
                                   float* out)
{
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c_out = n % OUT_SHAPE[1];
    n /= OUT_SHAPE[1];

    if (n < OUT_SHAPE[0] && x_out < OUT_SHAPE[2] && y_out < OUT_SHAPE[3])
    {
        float val = 0;

        int x_in = x_out * strideX, y_in = y_out * strideY;
        if (padding == PaddingType::kSAME)
        {
            x_in -= (KER_SHAPE[0] - 1) / 2;
            y_in -= (KER_SHAPE[1] - 1) / 2;
        }

        for (int dx = x_in < 0 ? -x_in : 0; dx < KER_SHAPE[0]; ++dx)
        {
            if (x_in + dx >= IN_SHAPE[2]) break;
            for (int dy = y_in < 0 ? -y_in : 0; dy < KER_SHAPE[1]; ++dy)
            {
                if (y_in + dy >= IN_SHAPE[3]) break;
                for (int c_in = 0; c_in < KER_SHAPE[2]; ++c_in)
                    val += in[POS_4D(n, c_in, x_in + dx, y_in + dy, IN_SHAPE)] *
                           ker[POS_4D(dx, dy, c_in, c_out, KER_SHAPE)];
            }
        }

        out[POS_4D(n, c_out, x_out, y_out, OUT_SHAPE)] = val;
    }
}

template <PaddingType padding>
__global__ void conv2D_grad_x_nhwc_kernel(const float* ker, const float* outG,
                                          float* inG)
{
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c_out = n % OUT_SHAPE[3];
    n /= OUT_SHAPE[3];

    if (n < OUT_SHAPE[0] && x_out < OUT_SHAPE[1] && y_out < OUT_SHAPE[2])
    {
        int x_in = x_out * strideX, y_in = y_out * strideY;
        if (padding == PaddingType::kSAME)
        {
            x_in -= (KER_SHAPE[0] - 1) / 2;
            y_in -= (KER_SHAPE[1] - 1) / 2;
        }

        float outG_val = outG[POS_4D(n, x_out, y_out, c_out, OUT_SHAPE)];

        for (int dx = x_in < 0 ? -x_in : 0; dx < KER_SHAPE[0]; ++dx)
        {
            if (x_in + dx >= IN_SHAPE[1]) break;
            for (int dy = y_in < 0 ? -y_in : 0; dy < KER_SHAPE[1]; ++dy)
            {
                if (y_in + dy >= IN_SHAPE[2]) break;
                for (int c_in = 0; c_in < KER_SHAPE[2]; ++c_in)
                {
                    float val =
                        outG_val * ker[POS_4D(dx, dy, c_in, c_out, KER_SHAPE)];
                    atomicAdd(
                        &inG[POS_4D(n, x_in + dx, y_in + dy, c_in, IN_SHAPE)],
                        val);
                }
            }
        }
    }
}

template <PaddingType padding>
__global__ void conv2D_grad_k_nhwc_kernel(const float* in, const float* outG,
                                          float* kerG)
{
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c_out = n % OUT_SHAPE[3];
    n /= OUT_SHAPE[3];

    if (n < OUT_SHAPE[0] && x_out < OUT_SHAPE[1] && y_out < OUT_SHAPE[2])
    {
        int x_in = x_out * strideX, y_in = y_out * strideY;
        if (padding == PaddingType::kSAME)
        {
            x_in -= (KER_SHAPE[0] - 1) / 2;
            y_in -= (KER_SHAPE[1] - 1) / 2;
        }

        float outG_val = outG[POS_4D(n, x_out, y_out, c_out, OUT_SHAPE)];

        for (int dx = x_in < 0 ? -x_in : 0; dx < KER_SHAPE[0]; ++dx)
        {
            if (x_in + dx >= IN_SHAPE[1]) break;
            for (int dy = y_in < 0 ? -y_in : 0; dy < KER_SHAPE[1]; ++dy)
            {
                if (y_in + dy >= IN_SHAPE[2]) break;
                for (int c_in = 0; c_in < KER_SHAPE[2]; ++c_in)
                {
                    float val =
                        outG_val *
                        in[POS_4D(n, x_in + dx, y_in + dy, c_in, IN_SHAPE)];
                    atomicAdd(&kerG[POS_4D(dx, dy, c_in, c_out, KER_SHAPE)],
                              val);
                }
            }
        }
    }
}

template <PaddingType padding>
__global__ void conv2D_grad_x_nchw_kernel(const float* ker, const float* outG,
                                          float* inG)
{
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c_out = n % OUT_SHAPE[1];
    n /= OUT_SHAPE[1];

    if (n < OUT_SHAPE[0] && x_out < OUT_SHAPE[2] && y_out < OUT_SHAPE[3])
    {
        int x_in = x_out * strideX, y_in = y_out * strideY;
        if (padding == PaddingType::kSAME)
        {
            x_in -= (KER_SHAPE[0] - 1) / 2;
            y_in -= (KER_SHAPE[1] - 1) / 2;
        }

        float outG_val = outG[POS_4D(n, c_out, x_out, y_out, OUT_SHAPE)];

        for (int dx = x_in < 0 ? -x_in : 0; dx < KER_SHAPE[0]; ++dx)
        {
            if (x_in + dx >= IN_SHAPE[2]) break;
            for (int dy = y_in < 0 ? -y_in : 0; dy < KER_SHAPE[1]; ++dy)
            {
                if (y_in + dy >= IN_SHAPE[3]) break;
                for (int c_in = 0; c_in < KER_SHAPE[2]; ++c_in)
                {
                    float val =
                        outG_val * ker[POS_4D(dx, dy, c_in, c_out, KER_SHAPE)];
                    atomicAdd(
                        &inG[POS_4D(n, c_in, x_in + dx, y_in + dy, IN_SHAPE)],
                        val);
                }
            }
        }
    }
}

template <PaddingType padding>
__global__ void conv2D_grad_k_nchw_kernel(const float* in, const float* outG,
                                          float* kerG)
{
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z * blockDim.z + threadIdx.z;
    int c_out = n % OUT_SHAPE[1];
    n /= OUT_SHAPE[1];

    if (n < OUT_SHAPE[0] && x_out < OUT_SHAPE[2] && y_out < OUT_SHAPE[3])
    {
        int x_in = x_out * strideX, y_in = y_out * strideY;
        if (padding == PaddingType::kSAME)
        {
            x_in -= (KER_SHAPE[0] - 1) / 2;
            y_in -= (KER_SHAPE[1] - 1) / 2;
        }

        float outG_val = outG[POS_4D(n, c_out, x_out, y_out, OUT_SHAPE)];

        for (int dx = x_in < 0 ? -x_in : 0; dx < KER_SHAPE[0]; ++dx)
        {
            if (x_in + dx >= IN_SHAPE[2]) break;
            for (int dy = y_in < 0 ? -y_in : 0; dy < KER_SHAPE[1]; ++dy)
            {
                if (y_in + dy >= IN_SHAPE[3]) break;
                for (int c_in = 0; c_in < KER_SHAPE[2]; ++c_in)
                {
                    float val =
                        outG_val *
                        in[POS_4D(n, c_in, x_in + dx, y_in + dy, IN_SHAPE)];
                    atomicAdd(&kerG[POS_4D(dx, dy, c_in, c_out, KER_SHAPE)],
                              val);
                }
            }
        }
    }
}

}  // namespace

void runConv2DDevice(const float* x, const float* k, float* y,
                     const int* params, PaddingType padding,
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

    cudaMemcpyToSymbol(shapeParams, params, 14 * sizeof(int));

    if (dataFormat == DataFormat::kNHWC)
    {
        if (padding == PaddingType::kVALID)
            conv2D_nhwc_kernel<PaddingType::kVALID><<<GRID, BLOCK>>>(x, k, y);
        else  // padding == PaddingType::kSAME
            conv2D_nhwc_kernel<PaddingType::kSAME><<<GRID, BLOCK>>>(x, k, y);
    }
    else  // dataFormat == DataFormat::kNCHW
    {
        if (padding == PaddingType::kVALID)
            conv2D_nchw_kernel<PaddingType::kVALID><<<GRID, BLOCK>>>(x, k, y);
        else  // padding == PaddingType::kSAME
            conv2D_nchw_kernel<PaddingType::kSAME><<<GRID, BLOCK>>>(x, k, y);
    }
}

void runConv2DGradientDevice(const float* x, const float* k, const float* yG,
                             float* xG, float* kG, const int* params,
                             PaddingType padding, DataFormat dataFormat)
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

    cudaMemcpyToSymbol(shapeParams, params, 14 * sizeof(int));

    size_t size = params[0] * params[1] * params[2] * params[3];
    utils::fill(xG, size, 0.);

    size = params[8] * params[9] * params[10] * params[11];
    utils::fill(kG, size, 0.);

    if (dataFormat == DataFormat::kNHWC)
    {
        if (padding == PaddingType::kVALID)
        {
            conv2D_grad_x_nhwc_kernel<PaddingType::kVALID>
                <<<GRID, BLOCK>>>(k, yG, xG);
            conv2D_grad_k_nhwc_kernel<PaddingType::kVALID>
                <<<GRID, BLOCK>>>(x, yG, kG);
        }
        else  // padding == PaddingType::kSAME
        {
            conv2D_grad_x_nhwc_kernel<PaddingType::kSAME>
                <<<GRID, BLOCK>>>(k, yG, xG);
            conv2D_grad_k_nhwc_kernel<PaddingType::kSAME>
                <<<GRID, BLOCK>>>(x, yG, kG);
        }
    }
    else  // dataFormat == DataFormat::kNCHW
    {
        if (padding == PaddingType::kVALID)
        {
            conv2D_grad_x_nchw_kernel<PaddingType::kVALID>
                <<<GRID, BLOCK>>>(k, yG, xG);
            conv2D_grad_k_nchw_kernel<PaddingType::kVALID>
                <<<GRID, BLOCK>>>(x, yG, kG);
        }
        else  // padding == PaddingType::kSAME
        {
            conv2D_grad_x_nchw_kernel<PaddingType::kSAME>
                <<<GRID, BLOCK>>>(k, yG, xG);
            conv2D_grad_k_nchw_kernel<PaddingType::kSAME>
                <<<GRID, BLOCK>>>(x, yG, kG);
        }
    }
}

#undef IN_SHAPE
#undef OUT_SHAPE
#undef KER_SHAPE
#undef strideX
#undef strideY

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
