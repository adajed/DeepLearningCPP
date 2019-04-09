#include "layers/convolution.h"

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
/* // info = [N, C_IN, X_IN, Y_IN, C_OUT, X_OUT, Y_OUT, X_KER, Y_KER, X_STR, Y_STR] */

/* #define N info[0] */
/* #define C_IN info[1] */
/* #define X_IN info[2] */
/* #define Y_IN info[3] */
/* #define C_OUT info[4] */
/* #define X_OUT info[5] */
/* #define Y_OUT info[6] */
/* #define X_KER info[7] */
/* #define Y_KER info[8] */
/* #define X_STR info[9] */
/* #define Y_STR info[10] */

/* template <PaddingType padding> */
/* __global__ void convKernel(const float* xArr, const float* kArr, float* yArr, */
/*                            const int* info) */
/* { */
/*     size_t outPos = blockIdx.x * blockDim.x + threadIdx.x; */
/*     size_t n = outPos; */

/*     int y = n % Y_OUT; */
/*     n /= Y_OUT; */
/*     int x = n % X_OUT; */
/*     n /= X_OUT; */
/*     int c = n % C_OUT; */
/*     n /= C_OUT; */

/*     if (n < N) */
/*     { */
/*         xArr += n * C_IN * X_IN * Y_IN; */
/*         kArr += c * C_IN * X_KER * Y_KER; */

/*         x *= X_STR; */
/*         y *= Y_STR; */
/*         if (padding == PaddingType::kSAME) */
/*         { */
/*             x -= (X_KER - 1) / 2; */
/*             y -= (Y_KER - 1) / 2; */
/*         } */

/*         float v = 0.; */
/*         for (int dc = 0; dc < C_IN; ++dc) */
/*         { */
/*             for (int dx = x < 0 ? -x : 0; dx < X_KER; ++dx) */
/*             { */
/*                 if (x + dx >= X_IN) break; */
/*                 for (int dy = y < 0 ? -y : 0; dy < Y_KER; ++dy) */
/*                 { */
/*                     if (y + dy >= Y_IN) break; */
/*                     v += xArr[(x + dx) * Y_IN + y + dy] * kArr[dx * Y_KER + dy]; */
/*                 } */
/*             } */

/*             xArr += X_IN * Y_IN; */
/*             kArr += X_KER * Y_KER; */
/*         } */

/*         yArr[outPos] = v; */
/*     } */
/* } */

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

/* void runConv2DDevice(const float* x, const float* k, float* y, */
/*                                 size_t size, int* info, PaddingType padding) */
/* { */
/*     const int BLOCK_SIZE = 256; */
/*     const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE; */

/*     if (padding == PaddingType::kVALID) */
/*         convKernel<PaddingType::kVALID> */
/*             <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, k, y, info); */
/*     else  // padding == PaddingType::kSAME */
/*         convKernel<PaddingType::kSAME> */
/*             <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, k, y, info); */
/* } */

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

/* #undef N */
/* #undef C_IN */
/* #undef X_IN */
/* #undef Y_IN */
/* #undef C_OUT */
/* #undef X_OUT */
/* #undef Y_OUT */
/* #undef X_KER */
/* #undef Y_KER */
/* #undef X_STR */
/* #undef Y_STR */

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
