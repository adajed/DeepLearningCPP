#ifndef GRAPHDL_CORE_LAYERS_POOLING_HOST_H_
#define GRAPHDL_CORE_LAYERS_POOLING_HOST_H_

#include "pooling.h"

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

template <PaddingType padding>
void pool_max_nhwc(const float* in, float* out, const std::vector<int>& inShape,
                   const std::vector<int>& outShape,
                   const std::vector<int>& kernel,
                   const std::vector<int>& strides)
{
#define POS_IN(n, x, y, c) \
    ((((n)*inShape[0] + (x)) * inShape[1] + (y)) * inShape[2] + (c))
#define POS_OUT(n, x, y, c) \
    ((((n)*outShape[0] + (x)) * outShape[1] + (y)) * outShape[2] + (c))

    for (int n = 0; n < outShape[0]; ++n)
        for (int x = 0; x < outShape[1]; ++x)
            for (int y = 0; y < outShape[2]; ++y)
                for (int c = 0; c < outShape[3]; ++c)
                {
                    int x2 = x * strides[0];
                    int y2 = y * strides[1];
                    if (padding == PaddingType::kSAME)
                    {
                        x2 -= (kernel[0] - 1) / 2;
                        y2 -= (kernel[1] - 1) / 2;
                    }

                    float val = 0;
                    if (x2 >= 0 && y2 >= 0) val = in[POS_IN(n, x2, y2, c)];

                    for (int iX = x2 > 0 ? x2 : 0; iX < x2 + kernel[0]; ++iX)
                    {
                        if (iX >= inShape[1])
                        {
                            val = val > 0 ? val : 0;
                            break;
                        }
                        for (int iY = y2 > 0 ? y2 : 0; iY < y2 + kernel[1];
                             ++iY)
                        {
                            if (iY >= inShape[2])
                            {
                                val = val > 0 ? val : 0;
                                break;
                            }

                            float f = in[POS_IN(n, iX, iY, c)];
                            val = val > f ? val : f;
                        }
                    }

                    out[POS_OUT(n, x, y, c)] = val;
                }

#undef POS_IN
#undef POS_OUT
}

template <PaddingType padding>
void pool_avg_nhwc(const float* in, float* out, const std::vector<int>& inShape,
                   const std::vector<int>& outShape,
                   const std::vector<int>& kernel,
                   const std::vector<int>& strides)
{
#define POS_IN(n, x, y, c) \
    ((((n)*inShape[0] + (x)) * inShape[1] + (y)) * inShape[2] + (c))
#define POS_OUT(n, x, y, c) \
    ((((n)*outShape[0] + (x)) * outShape[1] + (y)) * outShape[2] + (c))

    for (int n = 0; n < outShape[0]; ++n)
        for (int x = 0; x < outShape[1]; ++x)
            for (int y = 0; y < outShape[2]; ++y)
                for (int c = 0; c < outShape[3]; ++c)
                {
                    int x2 = x * strides[0];
                    int y2 = y * strides[1];
                    if (padding == PaddingType::kSAME)
                    {
                        x2 -= (kernel[0] - 1) / 2;
                        y2 -= (kernel[1] - 1) / 2;
                    }

                    float val = 0;
                    for (int iX = x2 > 0 ? x2 : 0; iX < x2 + kernel[0]; ++iX)
                    {
                        if (iX >= inShape[1]) break;
                        for (int iY = y2 > 0 ? y2 : 0; iY < y2 + kernel[1];
                             ++iY)
                        {
                            if (iY >= inShape[2]) break;
                            val += in[POS_IN(n, iX, iY, c)];
                        }
                    }

                    out[POS_OUT(n, x, y, c)] = val / (kernel[0] * kernel[1]);
                }

#undef POS_IN
#undef POS_OUT
}

template <PaddingType padding>
void pool_max_nchw(const float* in, float* out, const std::vector<int>& inShape,
                   const std::vector<int>& outShape,
                   const std::vector<int>& kernel,
                   const std::vector<int>& strides)
{
#define POS_IN(n, c, x, y) \
    ((((n)*inShape[0] + (c)) * inShape[1] + (x)) * inShape[2] + (y))
#define POS_OUT(n, c, x, y) \
    ((((n)*outShape[0] + (c)) * outShape[1] + (x)) * outShape[2] + (y))

    for (int n = 0; n < outShape[0]; ++n)
        for (int c = 0; c < outShape[1]; ++c)
            for (int x = 0; x < outShape[2]; ++x)
                for (int y = 0; y < outShape[3]; ++y)
                {
                    int x2 = x * strides[0];
                    int y2 = y * strides[1];
                    if (padding == PaddingType::kSAME)
                    {
                        x2 -= (kernel[0] - 1) / 2;
                        y2 -= (kernel[1] - 1) / 2;
                    }

                    float val = 0;
                    if (x2 >= 0 && y2 >= 0) val = in[POS_IN(n, c, x2, y2)];

                    for (int iX = x2 > 0 ? x2 : 0; iX < x2 + kernel[0]; ++iX)
                    {
                        if (iX >= inShape[2])
                        {
                            val = val > 0 ? val : 0;
                            break;
                        }
                        for (int iY = y2 > 0 ? y2 : 0; iY < y2 + kernel[1];
                             ++iY)
                        {
                            if (iY >= inShape[3])
                            {
                                val = val > 0 ? val : 0;
                                break;
                            }

                            float f = in[POS_IN(n, c, iX, iY)];
                            val = val > f ? val : f;
                        }
                    }

                    out[POS_OUT(n, x, y, c)] = val;
                }

#undef POS_IN
#undef POS_OUT
}

template <PaddingType padding>
void pool_avg_nchw(const float* in, float* out, const std::vector<int>& inShape,
                   const std::vector<int>& outShape,
                   const std::vector<int>& kernel,
                   const std::vector<int>& strides)
{
#define POS_IN(n, c, x, y) \
    ((((n)*inShape[0] + (c)) * inShape[1] + (x)) * inShape[2] + (y))
#define POS_OUT(n, c, x, y) \
    ((((n)*outShape[0] + (c)) * outShape[1] + (x)) * outShape[2] + (y))

    for (int n = 0; n < outShape[0]; ++n)
        for (int c = 0; c < outShape[1]; ++c)
            for (int x = 0; x < outShape[2]; ++x)
                for (int y = 0; y < outShape[3]; ++y)
                {
                    int x2 = x * strides[0];
                    int y2 = y * strides[1];
                    if (padding == PaddingType::kSAME)
                    {
                        x2 -= (kernel[0] - 1) / 2;
                        y2 -= (kernel[1] - 1) / 2;
                    }

                    float val = 0;
                    for (int iX = x2 > 0 ? x2 : 0; iX < x2 + kernel[0]; ++iX)
                    {
                        if (iX >= inShape[2]) break;
                        for (int iY = y2 > 0 ? y2 : 0; iY < y2 + kernel[1];
                             ++iY)
                        {
                            if (iY >= inShape[3]) break;
                            val += in[POS_IN(n, c, iX, iY)];
                        }
                    }

                    out[POS_OUT(n, c, x, y)] = val / (kernel[0] * kernel[1]);
                }

#undef POS_IN
#undef POS_OUT
}

template <PaddingType padding>
void pool_grad_max_nhwc(const float* in, const float* out, const float* outG,
                        float* inG, const std::vector<int>& inShape,
                        const std::vector<int>& outShape,
                        const std::vector<int>& kernel,
                        const std::vector<int>& strides)
{
#define POS_IN(n, x, y, c) \
    ((((n)*inShape[0] + (x)) * inShape[1] + (y)) * inShape[2] + (c))
#define POS_OUT(n, x, y, c) \
    ((((n)*outShape[0] + (x)) * outShape[1] + (y)) * outShape[2] + (c))

    for (int i = 0; i < inShape[0] * inShape[1] * inShape[2] * inShape[3]; ++i)
        inG[i] = 0.;

    for (int n = 0; n < outShape[0]; ++n)
        for (int x = 0; x < outShape[1]; ++x)
            for (int y = 0; y < outShape[2]; ++y)
                for (int c = 0; c < outShape[3]; ++c)
                {
                    int x2 = x * strides[0], y2 = y * strides[1];
                    if (padding == PaddingType::kSAME)
                    {
                        x2 -= (kernel[0] - 1) / 2;
                        y2 -= (kernel[1] - 1) / 2;
                    }

                    float out_val = out[POS_OUT(n, x, y, c)];
                    float outG_val = outG[POS_OUT(n, x, y, c)];

                    for (int iX = x2 > 0 ? x2 : 0; iX < x2 + kernel[0]; ++iX)
                    {
                        if (iX >= inShape[1]) break;
                        for (int iY = y2 > 0 ? y2 : 0; iY < y2 + kernel[1];
                             ++iX)
                        {
                            if (iY >= inShape[2]) break;
                            if (in[POS_IN(n, iX, iY, c)] == out_val)
                                inG[POS_IN(n, iX, iY, c)] += outG_val;
                        }
                    }
                }

#undef POS_IN
#undef POS_OUT
}

template <PaddingType padding>
void pool_grad_avg_nhwc(const float* in, const float* out, const float* outG,
                        float* inG, const std::vector<int>& inShape,
                        const std::vector<int>& outShape,
                        const std::vector<int>& kernel,
                        const std::vector<int>& strides)
{
#define POS_IN(n, x, y, c) \
    ((((n)*inShape[0] + (x)) * inShape[1] + (y)) * inShape[2] + (c))
#define POS_OUT(n, x, y, c) \
    ((((n)*outShape[0] + (x)) * outShape[1] + (y)) * outShape[2] + (c))

    for (int i = 0; i < inShape[0] * inShape[1] * inShape[2] * inShape[3]; ++i)
        inG[i] = 0.;

    for (int n = 0; n < outShape[0]; ++n)
        for (int x = 0; x < outShape[1]; ++x)
            for (int y = 0; y < outShape[2]; ++y)
                for (int c = 0; c < outShape[3]; ++c)
                {
                    int x2 = x * strides[0], y2 = y * strides[1];
                    if (padding == PaddingType::kSAME)
                    {
                        x2 -= (kernel[0] - 1) / 2;
                        y2 -= (kernel[1] - 1) / 2;
                    }

                    float outG_val =
                        outG[POS_OUT(n, x, y, c)] / (kernel[0] * kernel[1]);

                    for (int iX = x2 > 0 ? x2 : 0; iX < x2 + kernel[0]; ++iX)
                    {
                        if (iX >= inShape[1]) break;
                        for (int iY = y2 > 0 ? y2 : 0; iY < y2 + kernel[1];
                             ++iX)
                        {
                            if (iY >= inShape[2]) break;
                            inG[POS_IN(n, iX, iY, c)] += outG_val;
                        }
                    }
                }

#undef POS_IN
#undef POS_OUT
}

template <PaddingType padding>
void pool_grad_max_nchw(const float* in, const float* out, const float* outG,
                        float* inG, const std::vector<int>& inShape,
                        const std::vector<int>& outShape,
                        const std::vector<int>& kernel,
                        const std::vector<int>& strides)
{
#define POS_IN(n, c, x, y) \
    ((((n)*inShape[0] + (c)) * inShape[1] + (x)) * inShape[2] + (y))
#define POS_OUT(n, c, x, y) \
    ((((n)*outShape[0] + (c)) * outShape[1] + (x)) * outShape[2] + (y))

    for (int i = 0; i < inShape[0] * inShape[1] * inShape[2] * inShape[3]; ++i)
        inG[i] = 0.;

    for (int n = 0; n < outShape[0]; ++n)
        for (int c = 0; c < outShape[1]; ++c)
            for (int x = 0; x < outShape[2]; ++x)
                for (int y = 0; y < outShape[3]; ++y)
                {
                    int x2 = x * strides[0], y2 = y * strides[1];
                    if (padding == PaddingType::kSAME)
                    {
                        x2 -= (kernel[0] - 1) / 2;
                        y2 -= (kernel[1] - 1) / 2;
                    }

                    float out_val = out[POS_OUT(n, c, x, y)];
                    float outG_val = outG[POS_OUT(n, c, x, y)];

                    for (int iX = x2 > 0 ? x2 : 0; iX < x2 + kernel[0]; ++iX)
                    {
                        if (iX >= inShape[2]) break;
                        for (int iY = y2 > 0 ? y2 : 0; iY < y2 + kernel[1];
                             ++iX)
                        {
                            if (iY >= inShape[3]) break;
                            if (in[POS_IN(n, c, iX, iY)] == out_val)
                                inG[POS_IN(n, c, iX, iY)] += outG_val;
                        }
                    }
                }

#undef POS_IN
#undef POS_OUT
}

template <PaddingType padding>
void pool_grad_avg_nchw(const float* in, const float* out, const float* outG,
                        float* inG, const std::vector<int>& inShape,
                        const std::vector<int>& outShape,
                        const std::vector<int>& kernel,
                        const std::vector<int>& strides)
{
#define POS_IN(n, c, x, y) \
    ((((n)*inShape[0] + (c)) * inShape[1] + (x)) * inShape[2] + (y))
#define POS_OUT(n, c, x, y) \
    ((((n)*outShape[0] + (c)) * outShape[1] + (x)) * outShape[2] + (y))

    for (int i = 0; i < inShape[0] * inShape[1] * inShape[2] * inShape[3]; ++i)
        inG[i] = 0.;

    for (int n = 0; n < outShape[0]; ++n)
        for (int c = 0; c < outShape[1]; ++c)
            for (int x = 0; x < outShape[2]; ++x)
                for (int y = 0; y < outShape[3]; ++y)
                {
                    int x2 = x * strides[0], y2 = y * strides[1];
                    if (padding == PaddingType::kSAME)
                    {
                        x2 -= (kernel[0] - 1) / 2;
                        y2 -= (kernel[1] - 1) / 2;
                    }

                    float outG_val =
                        outG[POS_OUT(n, c, x, y)] / (kernel[0] * kernel[1]);

                    for (int iX = x2 > 0 ? x2 : 0; iX < x2 + kernel[0]; ++iX)
                    {
                        if (iX >= inShape[2]) break;
                        for (int iY = y2 > 0 ? y2 : 0; iY < y2 + kernel[1];
                             ++iX)
                        {
                            if (iY >= inShape[3]) break;
                            inG[POS_IN(n, c, iX, iY)] += outG_val;
                        }
                    }
                }

#undef POS_IN
#undef POS_OUT
}

}  // namespace layers
}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_POOLING_HOST_H_
