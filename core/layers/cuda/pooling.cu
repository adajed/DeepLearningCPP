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
__device__ float pool2DReduceKernel(const float* in, int* inShape, int* kernel,
                                    int x, int y);

template <>
__device__ float pool2DReduceKernel<PoolingType::kMAX>(const float* in,
                                                       int* inShape,
                                                       int* kernel, int x,
                                                       int y)
{
    float ret = in[0];

    for (int iX = 0; iX < kernel[0]; ++iX)
    {
        if (x + iX >= inShape[2]) break;
        for (int iY = 0; iY < kernel[1]; ++iY)
        {
            if (y + iY >= inShape[3]) break;
            float val = in[iX * inShape[3] + iY];
            ret = ret > val ? ret : val;
        }
    }

    return ret;
}

template <>
__device__ float pool2DReduceKernel<PoolingType::kAVERAGE>(const float* in,
                                                           int* inShape,
                                                           int* kernel, int x,
                                                           int y)
{
    float ret = 0.;

    for (int iX = 0; iX < kernel[0]; ++iX)
    {
        if (x + iX >= inShape[2]) break;
        for (int iY = 0; iY < kernel[1]; ++iY)
        {
            if (y + iY >= inShape[3]) break;
            ret += in[iX * inShape[3] + iY];
        }
    }

    return ret / (kernel[0] * kernel[1]);
}

template <PoolingType pooling>
__global__ void poolKernel(const float* in, float* out, int* inShape,
                           int* outShape, int* kernel, int* s)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t outPos = id;

    int y = id % outShape[3];
    id /= outShape[3];
    int x = id % outShape[2];
    id /= outShape[2];
    int c = outShape[1];
    id /= outShape[1];
    int n = id;

    if (n < outShape[0])
    {
        size_t inPos =
            ((n * inShape[1] + c) * inShape[2] + x * s[0]) * inShape[3] +
            y * s[1];
        out[outPos] = pool2DReduceKernel<pooling>(in + inPos, inShape, kernel,
                                                  x * s[0], y * s[1]);
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
                ;
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
        outShape[2] = shape[2] / strides[0];
        outShape[3] = shape[3] / strides[1];
    }
    else  // padding == PaddingType::kSAME
    {
        outShape[2] = ceil(shape[2], strides[0]);
        outShape[3] = ceil(shape[3], strides[1]);
    }
    size_t size = outShape[0] * outShape[1] * outShape[2] * outShape[3];

    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (pooling == PoolingType::kMAX)
        poolKernel<PoolingType::kMAX><<<NUM_BLOCKS, BLOCK_SIZE>>>(
            x, y, shape, outShape, kernel, strides);
    else  // pooling == PoolingType::kAVERAGE
        poolKernel<PoolingType::kAVERAGE><<<NUM_BLOCKS, BLOCK_SIZE>>>(
            x, y, shape, outShape, kernel, strides);
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
        outShape[2] = shape[2] / strides[0];
        outShape[3] = shape[3] / strides[1];
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
