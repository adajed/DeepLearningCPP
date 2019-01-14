#ifndef GRAPHDL_CORE_LAYERS_POOLING_HOST_H_
#define GRAPHDL_CORE_LAYERS_POOLING_HOST_H_

namespace graphdl
{
namespace core
{
namespace layers
{
int ceil(int x, int y)
{
    return (x / y) + int(x % y > 0);
}

template <PoolingType pooling>
float pool_reduce(const float* in, const std::vector<int>& shape,
                  const std::vector<int>& kernel, int x, int y);

template <>
float pool_reduce<PoolingType::kMAX>(const float* in,
                                     const std::vector<int>& shape,
                                     const std::vector<int>& kernel, int x,
                                     int y)
{
    float val = in[0];
    for (int iX = 0; iX < kernel[0]; ++iX)
    {
        if (x + iX >= shape[2])
        {
            val = val > 0. ? val : 0.;
            break;
        }
        for (int iY = 0; iY < kernel[1]; ++iY)
        {
            if (y + iY >= shape[3])
            {
                val = val > 0. ? val : 0.;
                break;
            }
            float f = in[iX * shape[3] + iY];
            val = val > f ? val : f;
        }
    }

    return val;
}

template <>
float pool_reduce<PoolingType::kAVERAGE>(const float* in,
                                         const std::vector<int>& shape,
                                         const std::vector<int>& kernel, int x,
                                         int y)
{
    float val = 0.;

    for (int iX = 0; iX < kernel[0]; ++iX)
    {
        if (x + iX >= shape[2]) break;
        for (int iY = 0; iY < kernel[1]; ++iY)
        {
            if (y + iY >= shape[3]) break;
            val += in[iX * shape[3] + iY];
        }
    }

    return val / (kernel[0] * kernel[1]);
}

template <PoolingType pooling, PaddingType padding>
void pool(const float* in, float* out, const std::vector<int>& shape,
          const std::vector<int>& k, const std::vector<int>& s)
{
    int outShape[] = {shape[0], shape[1], 0, 0};
    if (padding == PaddingType::kVALID)
    {
        outShape[2] = ceil(shape[2] - k[0] + 1, s[0]);
        outShape[3] = ceil(shape[3] - k[1] + 1, s[1]);
    }
    else  // padding == PaddingType::kSAME
    {
        outShape[2] = ceil(shape[2], s[0]);
        outShape[3] = ceil(shape[3], s[1]);
    }

    for (int n = 0; n < outShape[0]; ++n)
        for (int c = 0; c < outShape[1]; ++c)
            for (int x = 0; x < outShape[2]; ++x)
                for (int y = 0; y < outShape[3]; ++y)
                {
                    size_t outPos = ((n * outShape[1] + c) * outShape[2] + x) *
                                        outShape[3] +
                                    y;
                    size_t inPos =
                        ((n * shape[1] + c) * shape[2] + x * s[0]) * shape[3] +
                        y * s[1];
                    out[outPos] = pool_reduce<pooling>(in + inPos, shape, k,
                                                       x * s[0], y * s[1]);
                }
}

template <PoolingType pooling>
void pool_gradient_reduce(const float* in, float out, float outG, float* inG,
                          const std::vector<int>& shape,
                          const std::vector<int>& kernel, int x, int y);

template <>
void pool_gradient_reduce<PoolingType::kMAX>(const float* in, float out,
                                             float outG, float* inG,
                                             const std::vector<int>& shape,
                                             const std::vector<int>& kernel,
                                             int x, int y)
{
    for (int iX = 0; iX < kernel[0]; ++iX)
    {
        if (x + iX >= shape[2]) break;
        for (int iY = 0; iY < kernel[1]; ++iY)
        {
            if (y + iY >= shape[3]) break;
            float val = in[iX * shape[3] + iY];
            if (val == out) inG[iX * shape[3] + iY] += outG;
        }
    }
}

template <>
void pool_gradient_reduce<PoolingType::kAVERAGE>(
    const float* /* in */, float /* out */, float outG, float* inG,
    const std::vector<int>& shape, const std::vector<int>& kernel, int x, int y)
{
    float grad = outG / float(kernel[0] * kernel[1]);
    for (int iX = 0; iX < kernel[0]; ++iX)
    {
        if (x + iX >= shape[2]) break;
        for (int iY = 0; iY < kernel[1]; ++iY)
        {
            if (y + iY >= shape[3]) break;
            inG[iX * shape[3] + iY] += grad;
        }
    }
}

template <PoolingType pooling, PaddingType padding>
void poolGradient(const float* in, const float* out, const float* outG,
                  float* inG, const std::vector<int>& shape,
                  const std::vector<int>& k, const std::vector<int>& s)
{
    int outShape[] = {shape[0], shape[1], 0, 0};
    if (padding == PaddingType::kVALID)
    {
        outShape[2] = ceil(shape[2] - k[0] + 1, s[0]);
        outShape[3] = ceil(shape[3] - k[1] + 1, s[1]);
    }
    else  // padding == PaddingType::kSAME
    {
        outShape[2] = ceil(shape[2], s[0]);
        outShape[3] = ceil(shape[3], s[1]);
    }

    size_t inSize = shape[0] * shape[1] * shape[2] * shape[3];
    for (size_t pos = 0; pos < inSize; ++pos) inG[pos] = 0.;

    for (int n = 0; n < outShape[0]; ++n)
        for (int c = 0; c < outShape[1]; ++c)
            for (int x = 0; x < outShape[2]; ++x)
                for (int y = 0; y < outShape[3]; ++y)
                {
                    size_t outPos = n * outShape[1] + c;
                    outPos = outPos * outShape[2] + x;
                    outPos = outPos * outShape[3] + y;
                    size_t inPos = n * shape[1] + c;
                    inPos = inPos * shape[2] + x * s[0];
                    inPos = inPos * shape[3] + y * s[1];
                    pool_gradient_reduce<pooling>(in + inPos, out[outPos],
                                                  outG[outPos], inG + inPos,
                                                  shape, k, x * s[0], y * s[1]);
                }
}

}  // namespace layers
}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_POOLING_HOST_H_
