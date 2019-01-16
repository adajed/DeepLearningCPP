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
int ceil(int x, int y)
{
    return (x / y) + int(x % y > 0);
}

template <PoolingType pooling>
__device__ float pool2DReduceKernel(const float* in, int* info, int x, int y);

template <>
__device__ float pool2DReduceKernel<PoolingType::kMAX>(const float* in,
                                                       int* info, int x, int y)
{
    float ret = 0;
    if (x >= 0 && y >= 0) ret = in[x * info[3] + y];

    for (int iX = x > 0 ? x : 0; iX < x + info[8]; ++iX)
    {
        if (iX >= info[2])
        {
            ret = ret > 0. ? ret : 0.;
            break;
        }
        for (int iY = y > 0 ? y : 0; iY < y + info[9]; ++iY)
        {
            if (iY >= info[3])
            {
                ret = ret > 0. ? ret : 0.;
                break;
            }
            float val = in[iX * info[3] + iY];
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

    for (int iX = x > 0 ? x : 0; iX < x + info[8] && iX < info[2]; ++iX)
    {
        for (int iY = y > 0 ? y : 0; iY < y + info[9] && iY < info[3]; ++iY)
        {
            ret += in[iX * info[3] + iY];
        }
    }

    return ret / (info[8] * info[9]);
}

template <PoolingType pooling>
__global__ void poolKernel_valid(const float* in, float* out, int* info)
{
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    size_t outPos = n;

    int y = n % info[7];
    n /= info[7];
    int x = n % info[6];
    n /= info[6];
    int c = n % info[5];
    n /= info[5];

    if (n < info[4])
    {
        size_t inPos = (n * info[1] + c) * info[2] * info[3];
        out[outPos] = pool2DReduceKernel<pooling>(in + inPos, info,
                                                  x * info[10], y * info[11]);
    }
}

template <PoolingType pooling>
__global__ void poolKernel_same(const float* in, float* out, int* info)
{
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    size_t outPos = n;

    int y = n % info[7];
    n /= info[7];
    int x = n % info[6];
    n /= info[6];
    int c = n % info[5];
    n /= info[5];

    if (n < info[4])
    {
        size_t inPos = (n * info[1] + c) * info[2] * info[3];
        out[outPos] = pool2DReduceKernel<pooling>(
            in + inPos, info, x * info[10] - (info[8] - 1) / 2,
            y * info[11] - (info[9] - 1) / 2);
    }
}

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

template <PoolingType pooling, PaddingType padding>
__global__ void poolGradientKernel(const float* in, const float* out,
                                   const float* outG, float* inG, int* info)
{
    size_t n = blockIdx.x * blockDim.x + threadIdx.x;
    size_t inPos = n;

    int y = n % info[3];
    n /= info[3];
    int x = n % info[2];
    n /= info[2];
    int c = n % info[1];
    n /= info[1];

    if (n < info[0])
    {
        inG[inPos] = 0.;
        int xOut = firstWindow<padding>(x, info[8], info[10]);
        int yOut;

        while (out2in<padding>(xOut, info[8], info[10]) <= x && xOut < info[6])
        {
            yOut = firstWindow<padding>(y, info[9], info[11]);
            while (out2in<padding>(yOut, info[9], info[11]) <= y &&
                   yOut < info[7])
            {
                size_t outPos = n * info[5] + c;
                outPos = outPos * info[6] + xOut;
                outPos = outPos * info[7] + yOut;

                if (pooling == PoolingType::kMAX)
                {
                    if (in[inPos] == out[outPos]) inG[inPos] += outG[outPos];
                }
                else  // pooling == PoolingType::kAVERAGE
                {
                    inG[inPos] += *((float*)(info + 12)) * outG[outPos];
                }

                yOut++;
            }
            xOut++;
        }
    }
}

}  // namespace

extern "C" void runPool2DDevice(const float* x, float* y, int* shape,
                                int* kernel, int* strides, PoolingType pooling,
                                PaddingType padding)
{
    int outShape[] = {shape[0], shape[1], 0, 0};
    if (padding == PaddingType::kVALID)
    {
        outShape[2] = ceil(shape[2] - kernel[0] + 1, strides[0]);
        outShape[3] = ceil(shape[3] - kernel[1] + 1, strides[1]);
    }
    else  // padding == PaddingType::kSAME
    {
        outShape[2] = ceil(shape[2], strides[0]);
        outShape[3] = ceil(shape[3], strides[1]);
    }
    size_t size = outShape[0] * outShape[1] * outShape[2] * outShape[3];

    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int* info;
    cudaMalloc((void**)&info, 12 * sizeof(int));
    cudaMemcpy(info, shape, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(info + 4, outShape, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(info + 8, kernel, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(info + 10, strides, 2 * sizeof(int), cudaMemcpyHostToDevice);

    if (pooling == PoolingType::kMAX)
    {
        if (padding == PaddingType::kVALID)
            poolKernel_valid<PoolingType::kMAX>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, info);
        else  // padding == PaddingType::kSAME
            poolKernel_same<PoolingType::kMAX>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, info);
    }
    else  // pooling == PoolingType::kAVERAGE
    {
        if (padding == PaddingType::kVALID)
            poolKernel_valid<PoolingType::kAVERAGE>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, info);
        else  // padding == PaddingType::kSAME
            poolKernel_same<PoolingType::kAVERAGE>
                <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, info);
    }

    cudaDeviceSynchronize();
    cudaFree(info);
}

extern "C" void runPool2DGradientDevice(const float* x, const float* y,
                                        const float* yG, float* xG, int* shape,
                                        int* kernel, int* strides,
                                        PoolingType pooling,
                                        PaddingType padding)
{
    int outShape[] = {shape[0], shape[1], 0, 0};
    if (padding == PaddingType::kVALID)
    {
        outShape[2] = ceil(shape[2] - kernel[0] + 1, strides[0]);
        outShape[3] = ceil(shape[3] - kernel[1] + 1, strides[1]);
    }
    else  // padding == PaddingType::kSAME
    {
        outShape[2] = ceil(shape[2], strides[0]);
        outShape[3] = ceil(shape[3], strides[1]);
    }
    size_t size = shape[0] * shape[1] * shape[2] * shape[3];

    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float grad = 1. / float(kernel[0] * kernel[1]);
    int* info;
    cudaMalloc((void**)&info, 12 * sizeof(int) + sizeof(float));
    cudaMemcpy(info, shape, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(info + 4, outShape, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(info + 8, kernel, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(info + 10, strides, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(info + 12, &grad, sizeof(float), cudaMemcpyHostToDevice);

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

    cudaDeviceSynchronize();
    cudaFree(info);
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl