#ifndef GRAPHLDL_CORE_LAYERS_CONVOLUTION_HOST_H_
#define GRAPHLDL_CORE_LAYERS_CONVOLUTION_HOST_H_

#include "pooling.h"

namespace graphdl
{
namespace core
{
namespace layers
{
#define POS_4D(x1, x2, x3, x4, shape) \
    ((((x1)*shape[1] + (x2)) * shape[2] + (x3)) * shape[3] + (x4))

template <PaddingType padding>
void conv2d_nhwc(const float* in, const float* ker, float* out,
                 const std::vector<int>& inShape,
                 const std::vector<int>& outShape,
                 const std::vector<int>& kernel,
                 const std::vector<int>& strides)
{
#define POS_IN(n, x, y, c) POS_4D(n, x, y, c, inShape)
#define POS_OUT(n, x, y, c) POS_4D(n, x, y, c, outShape)
#define POS_KER(x, y, c1, c2) POS_4D(x, y, c1, c2, kernel)

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

                    float val = 0.;

                    for (int iX = x2 < 0 ? -x2 : 0; iX < kernel[0]; ++iX)
                    {
                        if (x2 + iX >= inShape[1]) break;
                        for (int iY = y2 < 0 ? -y2 : 0; iY < kernel[1]; ++iY)
                        {
                            if (y2 + iY >= inShape[2]) break;
                            for (int cIn = 0; cIn < kernel[2]; ++cIn)
                            {
                                val += in[POS_IN(n, x2 + iX, y2 + iY, cIn)] *
                                       ker[POS_KER(iX, iY, cIn, c)];
                            }
                        }
                    }

                    out[POS_OUT(n, x, y, c)] = val;
                }

#undef POS_IN
#undef POS_OUT
#undef POS_KER
}

template <PaddingType padding>
void conv2d_nchw(const float* in, const float* ker, float* out,
                 const std::vector<int>& inShape,
                 const std::vector<int>& outShape,
                 const std::vector<int>& kernel,
                 const std::vector<int>& strides)
{
#define POS_IN(n, c, x, y) POS_4D(n, c, x, y, inShape)
#define POS_OUT(n, c, x, y) POS_4D(n, c, x, y, outShape)
#define POS_KER(x, y, c1, c2) POS_4D(x, y, c1, c2, kernel)

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

                    float val = 0.;

                    for (int cIn = 0; cIn < kernel[2]; ++cIn)
                    {
                        for (int iX = x2 < 0 ? -x2 : 0; iX < kernel[0]; ++iX)
                        {
                            if (x2 + iX >= inShape[2]) break;
                            for (int iY = y2 < 0 ? -y2 : 0; iY < kernel[1];
                                 ++iY)
                            {
                                if (y2 + iY >= inShape[3]) break;
                                val += in[POS_IN(n, cIn, x2 + iX, y2 + iY)] *
                                       ker[POS_KER(iX, iY, cIn, c)];
                            }
                        }
                    }

                    out[POS_OUT(n, c, x, y)] = val;
                }

#undef POS_IN
#undef POS_OUT
#undef POS_KER
}

template <PaddingType padding>
void conv2d_grad_nhwc(const float* in, const float* ker, const float* outG,
                      float* inG, float* kerG, const std::vector<int>& inShape,
                      const std::vector<int>& outShape,
                      const std::vector<int>& kernel,
                      const std::vector<int>& strides)
{
#define POS_IN(n, x, y, c) POS_4D(n, x, y, c, inShape)
#define POS_OUT(n, x, y, c) POS_4D(n, x, y, c, outShape)
#define POS_KER(x, y, c1, c2) POS_4D(x, y, c1, c2, kernel)

    for (int i = 0; i < inShape[0] * inShape[1] * inShape[2] * inShape[3]; ++i)
        inG[i] = 0.;
    for (int i = 0; i < kernel[0] * kernel[1] * kernel[2] * kernel[3]; ++i)
        kerG[i] = 0.;

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

                    float out_grad = outG[POS_OUT(n, x, y, c)];

                    for (int cIn = 0; cIn < kernel[2]; ++cIn)
                    {
                        for (int iX = x2 < 0 ? -x2 : 0; iX < kernel[0]; ++iX)
                        {
                            if (x2 + iX >= inShape[1]) break;
                            for (int iY = y2 < 0 ? -y2 : 0; iY < kernel[1];
                                 ++iY)
                            {
                                if (y2 + iY >= inShape[2]) break;
                                inG[POS_IN(n, x2 + iX, y2 + iY, cIn)] +=
                                    out_grad * ker[POS_KER(iX, iY, cIn, c)];
                                kerG[POS_KER(iX, iY, cIn, c)] +=
                                    out_grad *
                                    in[POS_IN(n, x2 + iX, y2 + iY, cIn)];
                            }
                        }
                    }
                }

#undef POS_IN
#undef POS_OUT
#undef POS_KER
}

template <PaddingType padding>
void conv2d_grad_nchw(const float* in, const float* ker, const float* outG,
                      float* inG, float* kerG, const std::vector<int>& inShape,
                      const std::vector<int>& outShape,
                      const std::vector<int>& kernel,
                      const std::vector<int>& strides)
{
#define POS_IN(n, c, x, y) POS_4D(n, c, x, y, inShape)
#define POS_OUT(n, c, x, y) POS_4D(n, c, x, y, outShape)
#define POS_KER(x, y, c1, c2) POS_4D(x, y, c1, c2, kernel)

    for (int i = 0; i < inShape[0] * inShape[1] * inShape[2] * inShape[3]; ++i)
        inG[i] = 0.;
    for (int i = 0; i < kernel[0] * kernel[1] * kernel[2] * kernel[3]; ++i)
        kerG[i] = 0.;

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

                    float out_grad = outG[POS_OUT(n, c, x, y)];

                    for (int cIn = 0; cIn < kernel[2]; ++cIn)
                    {
                        for (int iX = x2 < 0 ? -x2 : 0; iX < kernel[0]; ++iX)
                        {
                            if (x2 + iX >= inShape[2]) break;
                            for (int iY = y2 < 0 ? -y2 : 0; iY < kernel[1];
                                 ++iY)
                            {
                                if (y2 + iY >= inShape[3]) break;
                                inG[POS_IN(n, cIn, x2 + iX, y2 + iY)] +=
                                    out_grad * ker[POS_KER(iX, iY, cIn, c)];
                                kerG[POS_KER(iX, iY, cIn, c)] +=
                                    out_grad *
                                    in[POS_IN(n, cIn, x2 + iX, y2 + iY)];
                            }
                        }
                    }
                }

#undef POS_IN
#undef POS_OUT
#undef POS_KER
}

#undef POS_4D

}  // namespace layers
}  // namespace core
}  // namespace graphdl

#endif  // GRAPHLDL_CORE_LAYERS_CONVOLUTION_HOST_H_
