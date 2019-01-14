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
    float ret = in[0];

    for (int iX = 0; iX < info[8]; ++iX)
    {
        if (x + iX >= info[2])
        {
            ret = ret > 0. ? ret : 0.;
            break;
        }
        for (int iY = 0; iY < info[9]; ++iY)
        {
            if (y + iY >= info[3])
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

    for (int iX = 0; iX < info[8] && x + iX < info[2]; ++iX)
    {
        for (int iY = 0; iY < info[9] && y + iY < info[3]; ++iY)
        {
            ret += in[iX * info[3] + iY];
        }
    }

    return ret / (info[8] * info[9]);
}

template <PoolingType pooling>
__global__ void poolKernel(const float* in, float* out, int* info)
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
        size_t inPos = n * info[1] + c;
        inPos = inPos * info[2] + x * info[10];
        inPos = inPos * info[3] + y * info[11];
        out[outPos] = pool2DReduceKernel<pooling>(in + inPos, info,
                                                  x * info[10], y * info[11]);
    }
}

__device__ int firstWindow(int x, int k, int s)
{
    int c = (x - k) / s + 1;
    c = c > 0 ? c : 0;
    c = (x - s * c) / s + 1;
    return c > 0 ? c : 0;
}

template <PoolingType pooling>
__global__ void poolGradientKernel(const float* in, const float* out,
                                   const float* outG, float* inG, int* inShape,
                                   int* outShape, int* kernel, int* strides);

template <>
__global__ void poolGradientKernel<PoolingType::kMAX>(
    const float* in, const float* out, const float* outG, float* inG,
    int* inShape, int* outShape, int* kernel, int* strides)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t inPos = id;

    int y = id % inShape[3];
    id /= inShape[3];
    int x = id % inShape[2];
    id /= inShape[2];
    int c = id % inShape[1];
    id /= inShape[1];
    int n = id;

    if (n < inShape[0])
    {
        inG[inPos] = 0.;
        int xOut = firstWindow(x, kernel[0], strides[0]);
        int yOut;

        while (xOut * strides[0] <= x && xOut < outShape[2])
        {
            yOut = firstWindow(y, kernel[1], strides[1]);
            while (yOut * strides[1] <= y && yOut < outShape[3])
            {
                size_t outPos =
                    ((n * outShape[1] + c) * outShape[2] + xOut) * outShape[3] +
                    yOut;

                if (in[inPos] == out[outPos]) inG[inPos] += outG[outPos];

                yOut++;
                ;
            }
            xOut++;
        }
    }
}

template <>
__global__ void poolGradientKernel<PoolingType::kAVERAGE>(
    const float* in, const float* out, const float* outG, float* inG,
    int* inShape, int* outShape, int* kernel, int* strides)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t inPos = id;

    int y = id % inShape[3];
    id /= inShape[3];
    int x = id % inShape[2];
    id /= inShape[2];
    int c = id % inShape[1];
    id /= inShape[1];
    int n = id;

    if (n < inShape[0])
    {
        inG[inPos] = 0.;
        int xOut = firstWindow(x, kernel[0], strides[0]);
        int yOut;

        while (xOut * strides[0] <= x && xOut < outShape[2])
        {
            yOut = firstWindow(y, kernel[1], strides[1]);
            while (yOut * strides[1] <= y && yOut < outShape[3])
            {
                size_t outPos =
                    ((n * outShape[1] + c) * outShape[2] + xOut) * outShape[3] +
                    yOut;
                inG[inPos] += outG[outPos];

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
        poolKernel<PoolingType::kMAX><<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, info);
    else  // pooling == PoolingType::kAVERAGE
        poolKernel<PoolingType::kAVERAGE>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, y, info);

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

    if (pooling == PoolingType::kMAX)
        poolGradientKernel<PoolingType::kMAX><<<NUM_BLOCKS, BLOCK_SIZE>>>(
            x, y, yG, xG, shape, outShape, kernel, strides);
    else  // pooling == PoolingType::kAVERAGE
        poolGradientKernel<PoolingType::kAVERAGE><<<NUM_BLOCKS, BLOCK_SIZE>>>(
            x, y, yG, xG, shape, outShape, kernel, strides);
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
