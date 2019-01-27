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
// info = [N, C_IN, X_IN, Y_IN, C_OUT, X_OUT, Y_OUT, X_KER, Y_KER, X_STR, Y_STR]

#define N info[0]
#define C_IN info[1]
#define X_IN info[2]
#define Y_IN info[3]
#define C_OUT info[4]
#define X_OUT info[5]
#define Y_OUT info[6]
#define X_KER info[7]
#define Y_KER info[8]
#define X_STR info[9]
#define Y_STR info[10]

template <PaddingType padding>
__global__ void convKernel(const float* xArr, const float* kArr, float* yArr,
                           const int* info)
{
    size_t outPos = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = outPos;

    int y = n % Y_OUT;
    n /= Y_OUT;
    int x = n % X_OUT;
    n /= X_OUT;
    int c = n % C_OUT;
    n /= C_OUT;

    if (n < N)
    {
        xArr += n * C_IN * X_IN * Y_IN;
        kArr += c * C_IN * X_KER * Y_KER;

        x *= X_STR;
        y *= Y_STR;
        if (padding == PaddingType::kSAME)
        {
            x -= (X_KER - 1) / 2;
            y -= (Y_KER - 1) / 2;
        }

        float v = 0.;
        for (int dc = 0; dc < C_IN; ++dc)
        {
            for (int dx = x < 0 ? -x : 0; dx < X_KER; ++dx)
            {
                if (x + dx >= X_IN) break;
                for (int dy = y < 0 ? -y : 0; dy < Y_KER; ++dy)
                {
                    if (y + dy >= Y_IN) break;
                    v += xArr[(x + dx) * Y_IN + y + dy] * kArr[dx * Y_KER + dy];
                }
            }

            xArr += X_IN * Y_IN;
            kArr += X_KER * Y_KER;
        }

        yArr[outPos] = v;
    }
}

}  // namespace

extern "C" void runConv2DDevice(const float* x, const float* k, float* y,
                                size_t size, int* info, PaddingType padding)
{
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (padding == PaddingType::kVALID)
        convKernel<PaddingType::kVALID>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, k, y, info);
    else  // padding == PaddingType::kSAME
        convKernel<PaddingType::kSAME>
            <<<NUM_BLOCKS, BLOCK_SIZE>>>(x, k, y, info);
}

extern "C" void runConv2DGradientDevice(const float* x, const float* k,
                                        const float* yG, float* xG, float* kG,
                                        int* shape, int* kernel, int* strides,
                                        PaddingType padding)
{
}

extern "C" void initializeConvGpuParams(void** dest, int* inShape,
                                        int* kerShape, int* outShape,
                                        int* strides)
{
    int* ptr;
    cudaMalloc((void**)&ptr, 11 * sizeof(int));
    cudaMemcpy(ptr, inShape, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr + 4, outShape + 1, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr + 7, kerShape + 2, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr + 9, strides, 2 * sizeof(int), cudaMemcpyHostToDevice);

    (*dest) = (void*)ptr;
}

#undef N
#undef C_IN
#undef X_IN
#undef Y_IN
#undef C_OUT
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
