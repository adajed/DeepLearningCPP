#include "layers/convolution.h"
#include "layers/cuda/macros.h"

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

/* template <PaddingType padding> */
/* __global__ void convGradientKernel_X(const float* kArr, const float* yGArr, */
/*                                      float* xGArr, int* info) */
/* { */
/*     size_t inPos = blockIdx.x * blockDim.x + threadIdx.x; */
/*     size_t n = inPos; */

/*     int y = n % Y_IN; */
/*     n /= Y_IN; */
/*     int x = n % X_IN; */
/*     n /= X_IN; */
/*     int c = n % C_IN; */
/*     n /= C_IN; */

/*     if (n < N) */
/*     { */
/*         xGArr[inPos] = 0.; */
/*         int xOut = firstWindow<padding>(x, X_KER, X_STR); */
/*         int yOut; */

/*         while (out2in<padding>(xOut, X_KER, X_STR) <= x && xOut < X_OUT) */
/*         { */
/*             yOut = firstWindow<padding>(y, Y_KER, Y_STR); */
/*             while (out2in<padding>(yOut, Y_KER, Y_STR) <= y && yOut < Y_OUT) */
/*             { */
/*                 for (int cOut = 0; cOut < C_OUT; ++cOut) */
/*                 { */
/*                     size_t outPos = n * C_OUT + cOut; */
/*                     outPos = outPos * X_OUT + xOut; */
/*                     outPos = outPos * Y_OUT + yOut; */
/*                     size_t kerPos = cOut * C_IN + c; */
/*                     kerPos = kerPos * X_KER + x - */
/*                              out2in<padding>(xOut, X_KER, X_STR); */
/*                     kerPos = kerPos * Y_KER + y - */
/*                              out2in<padding>(yOut, Y_KER, Y_STR); */
/*                     xGArr[inPos] += yGArr[outPos] * kArr[kerPos]; */
/*                 } */
/*                 yOut++; */
/*             } */
/*             xOut++; */
/*         } */
/*     } */
/* } */

/* template <PaddingType padding> */
/* __global__ void convGradientKernel_K(const float* xArr, const float* yGArr, */
/*                                      float* kGArr, int* info) */
/* { */
/*     size_t kerPos = blockIdx.x * blockDim.x + threadIdx.x; */
/*     size_t cOut = kerPos; */

/*     int yKer = cOut % Y_KER; */
/*     cOut /= Y_KER; */
/*     int xKer = cOut % X_KER; */
/*     cOut /= X_KER; */
/*     int cIn = cOut % C_IN; */
/*     cOut /= C_IN; */

/*     if (cOut < C_OUT) */
/*     { */
/*         kGArr[kerPos] = 0.; */
/*         for (int n = 0; n < N; ++n) */
/*         { */
/*             for (int xOut = 0; xOut < X_OUT; ++xOut) */
/*             { */
/*                 int xIn = out2in<padding>(xOut, X_KER, X_STR) + xKer; */
/*                 if (xIn < 0) continue; */
/*                 if (xIn >= X_IN) break; */
/*                 for (int yOut = 0; yOut < Y_OUT; ++yOut) */
/*                 { */
/*                     int yIn = out2in<padding>(yOut, Y_KER, Y_STR) + yKer; */
/*                     if (yIn < 0) continue; */
/*                     if (yIn >= Y_IN) break; */

/*                     size_t inPos = n * C_IN + cIn; */
/*                     inPos = inPos * X_IN + xIn; */
/*                     inPos = inPos * Y_IN + yIn; */
/*                     size_t outPos = n * C_OUT + cOut; */
/*                     outPos = outPos * X_OUT + xOut; */
/*                     outPos = outPos * Y_OUT + yOut; */
/*                     kGArr[kerPos] += xArr[inPos] * yGArr[outPos]; */
/*                 } */
/*             } */
/*         } */
/*     } */
/* } */

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

/* void runConv2DGradientDevice(const float* x, const float* k, */
/*                                         const float* yG, float* xG, float* kG, */
/*                                         size_t xSize, size_t kSize, int* info, */
/*                                         PaddingType padding) */
/* { */
/*     const int BLOCK_SIZE = 256; */
/*     const int NUM_BLOCKS_X = (xSize + BLOCK_SIZE - 1) / BLOCK_SIZE; */
/*     const int NUM_BLOCKS_K = (kSize + BLOCK_SIZE - 1) / BLOCK_SIZE; */

/*     if (padding == PaddingType::kVALID) */
/*     { */
/*         convGradientKernel_X<PaddingType::kVALID> */
/*             <<<NUM_BLOCKS_X, BLOCK_SIZE>>>(k, yG, xG, info); */
/*         convGradientKernel_K<PaddingType::kVALID> */
/*             <<<NUM_BLOCKS_K, BLOCK_SIZE>>>(x, yG, kG, info); */
/*     } */
/*     else  // padding == PaddingType::kSAME */
/*     { */
/*         convGradientKernel_X<PaddingType::kSAME> */
/*             <<<NUM_BLOCKS_X, BLOCK_SIZE>>>(k, yG, xG, info); */
/*         convGradientKernel_K<PaddingType::kSAME> */
/*             <<<NUM_BLOCKS_K, BLOCK_SIZE>>>(x, yG, kG, info); */
/*     } */
/* } */

/* extern "C" void initializeConvGpuParams(void* dest, int* inShape, int* kerShape, */
/*                                         int* outShape, int* strides) */
/* { */
/*     int* ptr = (int*)dest; */
/*     cudaMemcpy(ptr, inShape, 4 * sizeof(int), cudaMemcpyHostToDevice); */
/*     cudaMemcpy(ptr + 4, outShape + 1, 3 * sizeof(int), cudaMemcpyHostToDevice); */
/*     cudaMemcpy(ptr + 7, kerShape + 2, 2 * sizeof(int), cudaMemcpyHostToDevice); */
/*     cudaMemcpy(ptr + 9, strides, 2 * sizeof(int), cudaMemcpyHostToDevice); */
/* } */

#undef IN_SHAPE
#undef OUT_SHAPE
#undef KER_SHAPE
#undef strideX
#undef strideY
}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
