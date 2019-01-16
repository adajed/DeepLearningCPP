#include "abstractTensor.h"
#include "convolution.h"
#include "graph.h"

#include <cassert>

namespace graphdl
{
namespace core
{
namespace layers
{
namespace
{
int ceil(int x, int y)
{
    return x / y + int(x % y > 0);
}

Tensor::SPtr createOutput(const Tensor::SPtr& t,
                          const Tensor::SPtr& k,
                          const std::vector<int>& s,
                          PaddingType padding)
{
    TensorShape shape = t->getShape();
    TensorShape kernel = k->getShape();

    shape[1] = kernel[0];
    if (padding == PaddingType::kVALID)
    {
        shape[2] = ceil(shape[2] - kernel[2] + 1, s[0]);
        shape[3] = ceil(shape[3] - kernel[3] + 1, s[1]);
    }
    else  // padding == PaddingType::kSAME
    {
        shape[2] = ceil(shape[2], s[0]);
        shape[3] = ceil(shape[3], s[1]);
    }

    return createTensor("", shape, t->getType());
}

template<PaddingType padding>
void conv2D(const float* x, const float* k, float* y,
            const std::vector<int>& shape,
            const std::vector<int>& kernel,
            const std::vector<int>& strides);

template<>
void conv2D<PaddingType::kVALID>(
            const float* x, const float* k, float* y,
            const std::vector<int>& shape,
            const std::vector<int>& kernel,
            const std::vector<int>& strides)
{
}

template<>
void conv2D<PaddingType::kSAME>(
            const float* x, const float* k, float* y,
            const std::vector<int>& shape,
            const std::vector<int>& kernel,
            const std::vector<int>& strides)
{
}

void runConv2DHost(const float* x, const float* k, float* y,
                   const std::vector<int>& shape,
                   const std::vector<int>& kernel,
                   const std::vector<int>& strides,
                   PaddingType padding)
{
    if (padding == PaddingType::kVALID)
        conv2D<PaddingType::kVALID>(
                x, k, y, shape, kernel, strides);
    else  // padding == PaddingType::kSAME
        conv2D<PaddingType::kSAME>(
                x, k, y, shape, kernel, strides);
}

void runConv2DGradientHost(const float* in, const float* ker,
                           const float* out, const float* outG,
                           float* inG, float* kerG,
                           const std::vector<int>& shape,
                           const std::vector<int>& kernel,
                           const std::vector<int>& strides,
                           PaddingType padding)
{
}

}

Conv2DLayer::Conv2DLayer(ID id, const Tensor::SPtr& t,
                       const Tensor::SPtr& kernel,
                       const std::vector<int>& strides,
                       PaddingType padding)
    : DifferentiableLayer(id, {t, kernel},
                          {createOutput(t, kernel, strides, padding)}),
      mStrides(strides),
      mPadding(padding)
{
    assert(t->getType() == kernel->getType());
}

Layer::TensorMap Conv2DLayer::gradients(Tensor::SPtr out, Tensor::SPtr outGrad)
{
    assert(mOutputs[0] == out);

    Tensor::SPtr in = mInputs[0].lock();
    Tensor::SPtr k = mInputs[1].lock();

    Layer::SPtr layer = createLayer<Conv2DGradientLayer>(
            in, k, out, outGrad, mStrides, mPadding);

    return {{in, layer->getOutputs()[0]},
            {k, layer->getOutputs()[1]}};
}

void Conv2DLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr inTensor = mInputs[0].lock();
    Tensor::SPtr kerTensor = mInputs[1].lock();
    inTensor->eval(inputs);
    kerTensor->eval(inputs);

    float* in = inTensor->getMemory().getValues();
    float* ker = kerTensor->getMemory().getValues();
    float* out = mOutputs[0]->getMemory().getValues();

    TensorShape tShape = inTensor->getShape();
    TensorShape kShape = kerTensor->getShape();
    std::vector<int> inShape(tShape.begin(), tShape.end());
    std::vector<int> kerShape(kShape.begin(), kShape.end());

    if (inTensor->getType() == MemoryType::kHOST_MEMORY)
        runConv2DHost(in, ker, out, inShape, kerShape, mStrides, mPadding);
#ifdef CUDA_AVAILABLE
    else
        cuda::runConv2DDevice();
#endif
}

Conv2DGradientLayer::Conv2DGradientLayer(
        ID id, const Tensor::SPtr& t, const Tensor::SPtr& k,
        const Tensor::SPtr& out, const Tensor::SPtr& outG,
        const std::vector<int>& strides, PaddingType padding)
    : Layer(id, {t, k, out, outG}, {createTensor("", t->getShape(), t->getType()),
                                    createTensor("", k->getShape(), k->getType())}),
      mStrides(strides),
      mPadding(padding)
{
}

void Conv2DGradientLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr inTensor = mInputs[0].lock();
    Tensor::SPtr kerTensor = mInputs[1].lock();
    Tensor::SPtr outTensor = mInputs[2].lock();
    Tensor::SPtr outGTensor = mInputs[3].lock();
    inTensor->eval(inputs);
    kerTensor->eval(inputs);
    outTensor->eval(inputs);
    outGTensor->eval(inputs);

    float* in = inTensor->getMemory().getValues();
    float* ker = kerTensor->getMemory().getValues();
    float* out = outTensor->getMemory().getValues();
    float* outG = outGTensor->getMemory().getValues();
    float* inG = mOutputs[0]->getMemory().getValues();
    float* kerG = mOutputs[1]->getMemory().getValues();

    TensorShape tShape = inTensor->getShape();
    TensorShape kShape = kerTensor->getShape();
    std::vector<int> inShape(tShape.begin(), tShape.end());
    std::vector<int> kerShape(kShape.begin(), kShape.end());

    if (inTensor->getType() == MemoryType::kHOST_MEMORY)
        runConv2DGradientHost(in, ker, out, outG, inG, kerG,
                              inShape, kerShape, mStrides, mPadding);
#ifdef CUDA_AVAILABLE
    else
        cuda::runConv2DGradientDevice();
#endif
}

}  // namespace layers

Tensor::SPtr convolution2D(const Tensor::SPtr& t,
                           const Tensor::SPtr& kernel,
                           const std::vector<int>& strides,
                           layers::PaddingType padding)
{
    if (t->getShape().size() != 4)
        throw std::runtime_error("conv2D: wrong input shape");
    if (kernel->getShape().size() != 4)
        throw std::runtime_error("conv2D: wrong kernel shape");
    if (strides.empty() || strides.size() > 2)
        throw std::runtime_error("conv2D: wrong strides");

    for (int d : strides)
        if (d <= 0)
            throw std::runtime_error("conv2D: stride dims must be positive");

    Layer::SPtr layer = createLayer<layers::Conv2DLayer>(t, kernel, strides, padding);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr conv2D(const ITensorPtr& tensor,
                  const ITensorPtr& kernel,
                  const std::vector<int>& strides,
                  const std::string& padding)
{
    core::layers::PaddingType p;
    if (padding == "valid" || padding == "VALID")
        p = core::layers::PaddingType::kVALID;
    else if (padding == "same" || padding == "SAME")
        p = core::layers::PaddingType::kSAME;
    else
        throw std::runtime_error(
            R"(maxPool2D: wrong padding, should be one of: "SAME", "VALID")");

    core::Tensor::SPtr t = core::castITensorPtr(tensor)->get();
    core::Tensor::SPtr k = core::castITensorPtr(kernel)->get();
    return core::makeAbstractTensor(core::convolution2D(t, k, strides, p));
}

}  // namespace graphdl
