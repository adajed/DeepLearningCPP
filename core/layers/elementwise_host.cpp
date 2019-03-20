#include "elementwise.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace
{
template <Elementwise elem>
float op(float x1, float x2);
template <>
float op<Elementwise::kADD>(float x1, float x2)
{
    return x1 + x2;
}
template <>
float op<Elementwise::kSUB>(float x1, float x2)
{
    return x1 - x2;
}
template <>
float op<Elementwise::kMUL>(float x1, float x2)
{
    return x1 * x2;
}
template <>
float op<Elementwise::kDIV>(float x1, float x2)
{
    return x1 / x2;
}

template <Elementwise elem>
float opGrad1(float x1, float x2);
template <>
float opGrad1<Elementwise::kADD>(float /* x1 */, float /* x2 */)
{
    return 1.;
}
template <>
float opGrad1<Elementwise::kSUB>(float /* x1 */, float /* x2 */)
{
    return 1.;
}
template <>
float opGrad1<Elementwise::kMUL>(float /* x1 */, float x2)
{
    return x2;
}
template <>
float opGrad1<Elementwise::kDIV>(float /* x1 */, float x2)
{
    return 1. / x2;
}

template <Elementwise elem>
float opGrad2(float x1, float x2);
template <>
float opGrad2<Elementwise::kADD>(float /* x1 */, float /* x2 */)
{
    return 1.;
}
template <>
float opGrad2<Elementwise::kSUB>(float /* x1 */, float /* x2 */)
{
    return -1.;
}
template <>
float opGrad2<Elementwise::kMUL>(float x1, float /* x2 */)
{
    return x1;
}
template <>
float opGrad2<Elementwise::kDIV>(float x1, float x2)
{
    return -x1 / (x2 * x2);
}

template <Elementwise elem>
void elementwise(const float* x1, size_t size1, const float* x2, size_t size2,
                 float* y)
{
    if (size1 < size2)
    {
        for (size_t i = 0; i < size1; ++i)
            for (size_t b = i; b < size2; b += size1)
                y[b] = op<elem>(x1[i], x2[b]);
    }
    else
    {
        for (size_t i = 0; i < size2; ++i)
            for (size_t b = i; b < size1; b += size2)
                y[b] = op<elem>(x1[b], x2[i]);
    }
}


template <Elementwise elem>
void elementwiseGradient(const float* x1, size_t size1, const float* x2,
                         size_t size2, const float* yG, float* x1G, float* x2G)
{
    if (size1 < size2)
    {
        for (size_t i = 0; i < size1; ++i)
        {
            x1G[i] = 0.;
            for (size_t pos = i; pos < size2; pos += size1)
            {
                x1G[i] += yG[pos] * opGrad1<elem>(x1[i], x2[pos]);
                x2G[pos] = yG[pos] * opGrad2<elem>(x1[i], x2[pos]);
            }
        }
    }
    else
    {
        for (size_t i = 0; i < size2; ++i)
        {
            x2G[i] = 0.;
            for (size_t pos = i; pos < size1; pos += size2)
            {
                x1G[pos] = yG[pos] * opGrad1<elem>(x1[pos], x2[i]);
                x2G[i] += yG[pos] * opGrad2<elem>(x1[pos], x2[i]);
            }
        }
    }
}

template <Elementwise elem>
void elementwiseFront(const float* x1, size_t size1, const float* x2, size_t size2,
                      float* y)
{
    if (size1 > size2)
    {
        size_t reduceSize = size1 / size2;
        for (size_t i = 0; i < size1; ++i)
            y[i] = op<elem>(x1[i], x2[i / reduceSize]);
    }
    else
    {
        size_t reduceSize = size2 / size1;
        for (size_t i = 0; i < size2; ++i)
            y[i] = op<elem>(x1[i / reduceSize], x2[i]);
    }
}


template <Elementwise elem>
void elementwiseFrontGradient(const float* x1, size_t size1, const float* x2,
                              size_t size2, const float* yG, float* x1G, float* x2G)
{
    if (size1 < size2)
    {
        for (size_t i1 = 0; i1 < size1; ++i1)
        {
            x1G[i1] = 0.;
            for (size_t i2 = 0; i2 < size2 / size1; ++i2)
            {
                x2G[i2] = yG[i2] * opGrad2<elem>(x1[i1], x2[i2]);
                x1G[i1] += yG[i2] * opGrad1<elem>(x1[i1], x2[i2]);
            }

            x2 += size2 / size1;
            x2G += size2 / size1;
            yG += size2 / size1;
        }
    }
    else
    {
        for (size_t i2 = 0; i2 < size2; ++i2)
        {
            x2G[i2] = 0.;
            for (size_t i1 = 0; i1 < size1 / size2; ++i1)
            {
                x1G[i1] = yG[i1] * opGrad1<elem>(x1[i1], x2[i2]);
                x2G[i2] += yG[i1] * opGrad2<elem>(x1[i1], x2[i2]);
            }

            x1 += size1 / size2;
            x1G += size1 / size2;
            yG += size1 / size2;
        }
    }
}


}  // namespace

void runElementwiseBackHost(const float* x1, size_t size1, const float* x2,
                        size_t size2, float* y, Elementwise op)
{
    switch (op)
    {
    case Elementwise::kADD:
        elementwise<Elementwise::kADD>(x1, size1, x2, size2, y);
        return;
    case Elementwise::kSUB:
        elementwise<Elementwise::kSUB>(x1, size1, x2, size2, y);
        return;
    case Elementwise::kMUL:
        elementwise<Elementwise::kMUL>(x1, size1, x2, size2, y);
        return;
    case Elementwise::kDIV:
        elementwise<Elementwise::kDIV>(x1, size1, x2, size2, y);
        return;
    }
}

void runElementwiseBackGradientHost(const float* x1, size_t size1, const float* x2,
                                size_t size2, const float* yG, float* x1G,
                                float* x2G, Elementwise op)
{
    switch (op)
    {
    case Elementwise::kADD:
        elementwiseGradient<Elementwise::kADD>(x1, size1, x2, size2, yG, x1G,
                                               x2G);
        return;
    case Elementwise::kSUB:
        elementwiseGradient<Elementwise::kSUB>(x1, size1, x2, size2, yG, x1G,
                                               x2G);
        return;
    case Elementwise::kMUL:
        elementwiseGradient<Elementwise::kMUL>(x1, size1, x2, size2, yG, x1G,
                                               x2G);
        return;
    case Elementwise::kDIV:
        elementwiseGradient<Elementwise::kDIV>(x1, size1, x2, size2, yG, x1G,
                                               x2G);
        return;
    }
}

void runElementwiseFrontHost(const float* x1, size_t size1, const float* x2,
                        size_t size2, float* y, Elementwise op)
{
    switch (op)
    {
    case Elementwise::kADD:
        elementwiseFront<Elementwise::kADD>(x1, size1, x2, size2, y);
        return;
    case Elementwise::kSUB:
        elementwiseFront<Elementwise::kSUB>(x1, size1, x2, size2, y);
        return;
    case Elementwise::kMUL:
        elementwiseFront<Elementwise::kMUL>(x1, size1, x2, size2, y);
        return;
    case Elementwise::kDIV:
        elementwiseFront<Elementwise::kDIV>(x1, size1, x2, size2, y);
        return;
    }
}

void runElementwiseFrontGradientHost(const float* x1, size_t size1, const float* x2,
                                size_t size2, const float* yG, float* x1G,
                                float* x2G, Elementwise op)
{
    switch (op)
    {
    case Elementwise::kADD:
        elementwiseFrontGradient<Elementwise::kADD>(
                x1, size1, x2, size2, yG, x1G, x2G);
        return;
    case Elementwise::kSUB:
        elementwiseFrontGradient<Elementwise::kSUB>(
                x1, size1, x2, size2, yG, x1G, x2G);
        return;
    case Elementwise::kMUL:
        elementwiseFrontGradient<Elementwise::kMUL>(
                x1, size1, x2, size2, yG, x1G, x2G);
        return;
    case Elementwise::kDIV:
        elementwiseFrontGradient<Elementwise::kDIV>(
                x1, size1, x2, size2, yG, x1G, x2G);
        return;
    }
}

}  // namespace layers
}  // namespace core
}  // namespace graphdl
