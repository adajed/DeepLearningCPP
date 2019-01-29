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
// info = [N, C, X_IN, Y_IN, X_OUT, Y_OUT, X_KER, Y_KER, X_STR, Y_STR]

#define N info[0]
#define C info[1]
#define X_IN info[2]
#define Y_IN info[3]
#define X_OUT info[4]
#define Y_OUT info[5]
#define X_KER info[6]
#define Y_KER info[7]
#define X_STR info[8]
#define Y_STR info[9]
// KER_SIZE_REC is 1. / (X_KER * Y_KER)
#define KER_SIZE_REC (*((float*)(info + 10)))

template <PaddingType padding>
__device__ int firstWindow(int x, int k, int s);

template <>
__device__ int firstWindow<PaddingType::kVALID>(int x, int k, int s)
{
    int c = x - k + 1;
    c = c > 0 ? c : 0;
    return c / s + int(c % s > 0);
}

template <>
__device__ int firstWindow<PaddingType::kSAME>(int x, int k, int s)
{
    int c = x + (k - 1) / 2 - k + 1;
    c = c > 0 ? c : 0;
    return c / s + int(c % s > 0);
}

template <PaddingType>
__device__ int out2in(int x, int k, int s);

template <>
__device__ int out2in<PaddingType::kVALID>(int x, int /* k */, int s)
{
    return x * s;
}

template <>
__device__ int out2in<PaddingType::kSAME>(int x, int k, int s)
{
    return x * s - (k - 1) / 2;
}

template <PoolingType pooling>
__device__ float pool2DReduceKernel(const float* in, int* info, int x, int y);

template <>
__device__ float pool2DReduceKernel<PoolingType::kMAX>(const float* in,
                                                       int* info, int x, int y)
{
    float ret = 0;
    if (x >= 0 && y >= 0) ret = in[x * Y_IN + y];

    for (int iX = x > 0 ? x : 0; iX < x + X_KER; ++iX)
    {
        if (iX >= X_IN)
        {
            ret = ret > 0. ? ret : 0.;
            break;
        }
        for (int iY = y > 0 ? y : 0; iY < y + Y_KER; ++iY)
        {
            if (iY >= Y_IN)
            {
                ret = ret > 0. ? ret : 0.;
                break;
            }
            float val = in[iX * Y_IN + iY];
            ret = ret > val ? ret : val;
        }
    }

    return ret;
}

template <>
__device__ float pool2DReduceKernel<PoolingType::kAVERAGE>(const float* in,
                                                           int* info, int x,
                                                           int y)
{
    float ret = 0.;

    for (int iX = x > 0 ? x : 0; iX < x + X_KER && iX < X_IN; ++iX)
    {
        for (int iY = y > 0 ? y : 0; iY < y + Y_KER && iY < Y_IN; ++iY)
        {
            ret += in[iX * Y_IN + iY];
        }
    }

    return ret * KER_SIZE_REC;
}

template <PoolingType pooling, PaddingType padding>
__global__ void poolKernel(const float* in, float* out, int* info)
{
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    size_t outPos = n;

    int y = n % Y_OUT;
    n /= Y_OUT;
    int x = n % X_OUT;
    n /= X_OUT;
    int c = n % C;
    n /= C;

    if (n < N)
    {
        size_t inPos = (n * C + c) * X_IN * Y_IN;
        out[outPos] = pool2DReduceKernel<pooling>(
            in + inPos, info, out2in<padding>(x, X_KER, X_STR),
            out2in<padding>(y, Y_KER, Y_STR));
    }
}

template <PoolingType pooling, PaddingType padding>
__global__ void poolGradientKernel(const float* in, const float* out,
                                   const float* outG, float* inG, int* info)
{
    size_t inPos = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = inPos;

    int y = n % Y_IN;
    n /= Y_IN;
    int x = n % X_IN;
    n /= X_IN;
    int c = n % C;
    n /= C;

    if (n < N)
    {
        inG[inPos] = 0.;
        int xOut = firstWindow<padding>(x, X_KER, X_STR);
        int yOut;

        while (out2in<padding>(xOut, X_KER, X_STR) <= x && xOut < X_OUT)
        {
            yOut = firstWindow<padding>(y, Y_KER, Y_STR);
            while (out2in<padding>(yOut, Y_KER, Y_STR) <= y && yOut < Y_OUT)
            {
                size_t outPos = n * C + c;
                outPos = outPos * X_OUT + xOut;
                outPos = outPos * Y_OUT + yOut;

                if (pooling == PoolingType::kMAX)
                {
                    if (in[inPos] == out[outPos]) inG[inPos] += outG[outPos];
                }
                else  // pooling == PoolingType::kAVERAGE
                {
                    inG[inPos] += KER_SIZE_REC * outG[outPos];
                }

                yOut++;
            }
            xOut++;
        }
    }
}

}  // namespace

extern "C" void runPool2DDevice(const float* x, float* y, int* info,
                                size_t size, PoolingType pooling,
                                PaddingType padding)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (pooling == PoolingType::kMAX)
    {
        if (padding == PaddingType::kVALID)
            poolKernel<PoolingType::kMAX, PaddingType::kVALID>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, info);
        else  // padding == PaddingType::kSAME
            poolKernel<PoolingType::kMAX, PaddingType::kSAME>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, info);
    }
    else  // pooling == PoolingType::kAVERAGE
    {
        if (padding == PaddingType::kVALID)
            poolKernel<PoolingType::kAVERAGE, PaddingType::kVALID>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, info);
        else  // padding == PaddingType::kSAME
            poolKernel<PoolingType::kAVERAGE, PaddingType::kSAME>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, info);
    }
}

extern "C" void runPool2DGradientDevice(const float* x, const float* y,
                                        const float* yG, float* xG, int* info,
                                        size_t size, PoolingType pooling,
                                        PaddingType padding)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (pooling == PoolingType::kMAX)
    {
        if (padding == PaddingType::kVALID)
            poolGradientKernel<PoolingType::kMAX, PaddingType::kVALID>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yG, xG, info);
        else  // padding == PaddingType::kSAME
            poolGradientKernel<PoolingType::kMAX, PaddingType::kSAME>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yG, xG, info);
    }
    else  // pooling == PoolingType::kAVERAGE
    {
        if (padding == PaddingType::kVALID)
            poolGradientKernel<PoolingType::kAVERAGE, PaddingType::kVALID>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yG, xG, info);
        else  // padding == PaddingType::kSAME
            poolGradientKernel<PoolingType::kAVERAGE, PaddingType::kSAME>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, yG, xG, info);
    }
}

extern "C" void initializePoolGpuParams(void* dest, int* inShape, int* kernel,
                                        int* strides, int* outShape)
{
    int* ptr = (int*)dest;
    float grad = 1. / float(kernel[0] * kernel[1]);
    cudaMemcpy(ptr, inShape, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr + 4, outShape + 2, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr + 6, kernel, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr + 8, strides, 2 * sizeof(int), cudaMemcpyHostToDevice);

    // necessary for gradient calculating
    cudaMemcpy(ptr + 10, &grad, sizeof(float), cudaMemcpyHostToDevice);
}

#undef N
#undef C
#undef X_IN
#undef Y_IN
#undef X_OUT
#undef Y_OUT
#undef X_KER
#undef Y_KER
#undef X_STR
#undef Y_STR

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
