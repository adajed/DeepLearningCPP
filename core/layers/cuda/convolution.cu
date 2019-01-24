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
// info = [in(N, C, X, Y), ker(outC, inC, kX, kY), out(N, C, X, Y), strides]
template <PaddingType padding>
__global__ void convKernel(const float* xArr, const float* kArr, float* yArr,
                           const int* info)
{
    size_t outPos = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = outPos;

    int y = n % info[11];
    n /= info[11];
    int x = n % info[10];
    n /= info[10];
    int c = n % info[9];
    n /= info[9];

    if (n < info[8])
    {
        xArr += n * info[1] * info[2] * info[3];
        kArr += c * info[5] * info[6] * info[7];

        x *= info[12];
        y *= info[13];
        if (padding == PaddingType::kSAME)
        {
            x -= (info[6] - 1) / 2;
            y -= (info[7] - 1) / 2;
        }

        float v = 0.;
        for (int dc = 0; dc < info[5]; ++dc)
        {
            for (int dx = x < 0 ? -x : 0; dx < info[6]; ++dx)
            {
                if (x + dx >= info[2]) break;
                for (int dy = y < 0 ? -y : 0; dy < info[7]; ++dy)
                {
                    if (y + dy >= info[3]) break;
                    v += xArr[(x + dx) * info[3] + y + dy] *
                         kArr[dx * info[7] + dy];
                }
            }

            xArr += info[2] * info[3];
            kArr += info[6] * info[7];
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
    cudaMalloc((void**)&ptr, 14 * sizeof(int));
    cudaMemcpy(ptr, inShape, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr + 4, kerShape, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr + 8, outShape, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ptr + 12, strides, 2 * sizeof(int), cudaMemcpyHostToDevice);

    (*dest) = (void*)ptr;
}

}  // namespace cuda
}  // namespace layers
}  // namespace core
}  // namespace graphdl
