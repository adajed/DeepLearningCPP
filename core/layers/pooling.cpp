#include "pooling.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace
{
Tensor::SPtr createPoolingOutput(Tensor::SPtr t, const std::vector<int>& kernel,
                                 const std::vector<int>& strides, PaddingType padding)
{
    TensorShape shape = t->getShape();
    std::vector<int> newShape;
    newShape.push_back(shape[0]);
    newShape.push_back(shape[1] / strides[0]);
    newShape.push_back(shape[2] / strides[1]);
    newShape.push_back(shape[3]);

    return createTensor("", newShape, t->getType());
}

int ceil(int x, int y)
{
    return (x / y) + int(x % y > 0);
}

template <PoolingType pooling>
void pool_reduce(const float* in, float* out,
                 const std::vector<int>& shape,
                 const std::vector<int>& kernel,
                 const std::vector<int>& strides,
                 int n, int x, int y, int c);

template <>
void pool_reduce<PoolingType::kMAX>(
    const float* in, float* out,
    const std::vector<int>& shape,
    const std::vector<int>& kernel,
    const std::vector<int>& strides,
    int n, int x, int y, int c)
{
    size_t pos = n * shape[1] * shape[2] * shape[3]
               + x * strides[0] * shape[2] * shape[3]
               + y * strides[1] * shape[3]
               + c;
    float val;

    for (int iX = 0; iX < kernel[0]; ++iX)
    {
        if (x * strides[0] + iX >= shape[1])
            break;
        for (int iY = 0; iY < kernel[1]; ++iY)
        {
            if (y * strides[1] + iY >= shape[2])
                break;

            pos = n * shape[1] * shape[2] * shape[3]
                + (x * strides[0] + iX) * shape[2] * shape[3]
                + (y * strides[1] + iY) * shape[3]
                + c;
            float f = in[pos];
            val = val > f ? val : f;
        }
    }

    pos = n * ceil(shape[1], s[0]) * ceil(shape[2], s[1]) * shape[3]
        + x * ceil(shape[2], s[1]) * shape[3]
        + y * shape[3]
        + c;
    out[pos] = val;
}

template <>
void pool_reduce<PoolingType::kAVERAGE>(
    const float* in, float* out,
    const std::vector<int>& shape,
    const std::vector<int>& kernel,
    const std::vector<int>& strides,
    int n, int x, int y, int c)
{
    size_t pos;
    float val = 0.;

    for (int iX = 0; iX < kernel[0]; ++iX)
    {
        if (x * strides[0] + iX >= shape[1])
            break;
        for (int iY = 0; iY < kernel[1]; ++iY)
        {
            if (y * strides[1] + iY >= shape[2])
                break;

            pos = n * shape[1] * shape[2] * shape[3]
                + (x * strides[0] + iX) * shape[2] * shape[3]
                + (y * strides[1] + iY) * shape[3]
                + c;
            val += in[pos];
        }
    }

    val /= kernel[0] * kernel[1];

    pos = n * ceil(shape[1], s[0]) * ceil(shape[2], s[1]) * shape[3]
        + x * ceil(shape[2], s[1]) * shape[3]
        + y * shape[3]
        + c;
    out[pos] = val;
}

template <PoolingType pooling, PaddingType padding>
void pool(const float* in, float* out, const std::vector<int>& shape,
          const std::vector<int>& k, const std::vector<int>& s);

template <PoolingType pooling>
void pool<PaddingType::kVALID>(const float* in, float* out, const std::vector<int>& shape,
                               const std::vector<int>& k, const std::vector<int>& s);
{
    int n = 0, c = 0, x = 0, y = 0;
    while (n < shape[0])
    {
        pool_reduce<pooling>(in, out, shape, k, s, n, x ,y, c);

        y++;
        if (y * s[1] + k[1] >= shape[2])
        {
            y = 0;
            x++;
            if (x * s[0] + k[0] >= shape[1])
            {
                x = 0;
                if ((c += 1) >= shape[3])
                {
                    c = 0;
                    n++;
                }
            }
        }
    }
}

template <PoolingType pooling>
void pool<PaddingType::kSAME>(const float* in, float* out, const std::vector<int>& shape,
                               const std::vector<int>& k, const std::vector<int>& s);
{
    int n = 0, c = 0, x = 0, y = 0;
    while (n < shape[0])
    {
        pool_reduce<pooling>(in, out, shape, k, s, n, x ,y, c);

        y++;
        if (y * s[1] >= shape[2])
        {
            y = 0;
            x++;
            if (x * s[0] >= shape[1])
            {
                x = 0;
                if ((c += 1) >= shape[3])
                {
                    c = 0;
                    n++;
                }
            }
        }
    }
}

void runPooling2DHost(const float* x, float* y, int kW, int kH, int sW, int sH,
                      PoolingType pooling, PaddingType padding)
{
}

void runPooling2DGradientHost()
{
}


}

Pooling2DLayer::Pooling2DLayer(ID id, const Tensor::SPtr& t, PoolingType pooling,
    const std::vector<int>& kernel, const std::vector<int>& strides, PaddingType padding)
    : DifferentiableLayer(id, {t}, {createPoolingOutput(t, kernel, strides, padding)}),
      mPooling(pooling),
      mKernelWindow(kernel),
      mStrides(strides),
      mPadding(padding)
{
    assert(kernel.size() == 2);
    assert(strides.size() == 2);
}

Layer::TensorMap Pooling2DLayer::gradients(Tensor::SPtr out, Tensor::SPtr outGrad)
{
    Tensor::SPtr input = mInputs[0].lock();
    Layer::SPtr layer = createLayer<Pooling2DGradientLayer>(
            input, out, outGrad, mPooling, mKernelWindow, mStrides, mPadding);
    return {{input, layer->getOutputs()[0]}};
}

void Pooling2DLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr inTensor = mInputs[0].lock();
    inTensor->eval(inputs);

    float* in = inTensor->getMemory().getValues();
    float* out = mOutputs[0]->getMemory().getvalues();

    if (inTensor->getType() == MemoryType::kHOST_MEMORY)
    {}
    else  // inTensor->getType() == MemoryType::kDEVICE_MEMORY
    {}
}

Pooling2DGradientLayer::Pooling2DGradientLayer(ID id, const Tensor::SPtr& t,
        const Tensor::SPtr& out, const Tensor::SPtr& outGrad,
        PoolingType pooling, const std::vector<int>& kernel,
        const std::vector<int>& strides, PaddingType padding)
    : Layer(id, {t, out, outGrad}, {createTensor("", t->getShape(), t->getType())}),
      mPooling(pooling),
      mKernelWindow(kernel),
      mStrides(strides),
      mPadding(padding)
{
    assert(kernel.size() == 2);
    assert(strides.size() == 2);
}

void Pooling2DGradientLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr outGradTensor = mInputs[2].lock();
    outGradTensor->eval(inputs);

    float* outGrad = outGradTensor->getMemory().getValues();

    if (outGradTensor->getType() == MemoryType::kHOST_MEMORY)
    {}
    else  // outGradTensor->getType() == MemoryType::kDEVICE_MEMORY
    {}
}

}  // namespace layers

Tensor::SPtr pooling2D(const Tensor::SPtr& t,
                       layers::PoolingType pooling,
                       const std::vector<int>& kernel,
                       const std::vector<int>& strides,
                       layers::PaddingType padding)
{
    if (t->getShape().size() != 4)
        throw std::runtime_error("pool2D: wrong input shape");

    Layer::SPtr layer = createLayer<layers::Pooling2DLayer>(
            t, pooling, kernel, strides, padding);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr maxPool2D(const ITensorPtr& tensor, const std::vector<int>& kernelSize,
                     const std::vector<int>& strides, const std::string& padding)
{
    core::layers::PaddingType p;
    if (padding == "valid" || padding == "VALID")
        p = core::layers::PaddingType::kVALID;
    if (padding == "same" || padding == "SAME")
        p = core::layers::PaddingType::kSAME;

    Tensor::SPtr t = core::castITensorPtr(tensor)->get();
    return core::makeAbstractTensor(core::pooling2D(
                t, core::layers::PoolingType::kMAX, kernelSize, strides, p));
}

ITensorPtr avgPool2D(const ITensorPtr& tensor, const std::vector<int>& kernelSize,
                     const std::vector<int>& strides, const std::string& padding)
{
    core::layers::PaddingType p;
    if (padding == "valid" || padding == "VALID")
        p = core::layers::PaddingType::kVALID;
    if (padding == "same" || padding == "SAME")
        p = core::layers::PaddingType::kSAME;

    Tensor::SPtr t = core::castITensorPtr(tensor)->get();
    return core::makeAbstractTensor(core::pooling2D(
                t, core::layers::PoolingType::kAVERAGE, kernelSize, strides, p));
}

}  // namespace graphld
