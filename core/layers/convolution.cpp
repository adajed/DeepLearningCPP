#include "convolution.h"

#include "abstractTensor.h"
#include "convolution_host.h"
#include "graph.h"

#include <cassert>
#include <utility>

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

Tensor::SPtr createOutput(const Tensor::SPtr& t, const Tensor::SPtr& k,
                          const std::vector<int>& s, PaddingType padding,
                          DataFormat dataFormat)
{
    TensorShape shape = t->getShape();
    TensorShape kernel = k->getShape();

    int x = dataFormat == DataFormat::kNHWC ? 1 : 2;
    int y = dataFormat == DataFormat::kNHWC ? 2 : 3;
    int c = dataFormat == DataFormat::kNHWC ? 3 : 1;

    shape[c] = kernel[3];
    if (padding == PaddingType::kVALID)
    {
        shape[x] = ceil(shape[x] - kernel[0] + 1, s[0]);
        shape[y] = ceil(shape[y] - kernel[1] + 1, s[1]);
    }
    else  // padding == PaddingType::kSAME
    {
        shape[x] = ceil(shape[x], s[0]);
        shape[y] = ceil(shape[y], s[1]);
    }

    return createTensor("", shape, t->getType());
}

void runConv2DHost(const float* x, const float* k, float* y,
                   const std::vector<int>& inShape,
                   const std::vector<int>& outShape,
                   const std::vector<int>& kernel,
                   const std::vector<int>& strides, PaddingType padding,
                   DataFormat dataFormat)
{
    if (dataFormat == DataFormat::kNHWC)
    {
        if (padding == PaddingType::kVALID)
            conv2d_nhwc<PaddingType::kVALID>(x, k, y, inShape, outShape, kernel,
                                             strides);
        else  // padding == PaddingType::kSAME
            conv2d_nhwc<PaddingType::kSAME>(x, k, y, inShape, outShape, kernel,
                                            strides);
    }
    else
    {
        if (padding == PaddingType::kVALID)
            conv2d_nchw<PaddingType::kVALID>(x, k, y, inShape, outShape, kernel,
                                             strides);
        else  // padding == PaddingType::kSAME
            conv2d_nchw<PaddingType::kSAME>(x, k, y, inShape, outShape, kernel,
                                            strides);
    }
}

void runConv2DGradientHost(const float* in, const float* ker, const float* out,
                           float* inG, float* kerG,
                           const std::vector<int>& inShape,
                           const std::vector<int>& outShape,
                           const std::vector<int>& kernel,
                           const std::vector<int>& strides, PaddingType padding,
                           DataFormat dataFormat)
{
    if (dataFormat == DataFormat::kNHWC)
    {
        if (padding == PaddingType::kVALID)
            conv2d_grad_nhwc<PaddingType::kVALID>(
                in, ker, out, inG, kerG, inShape, outShape, kernel, strides);
        else  // padding == PaddingType::kSAME
            conv2d_grad_nhwc<PaddingType::kSAME>(
                in, ker, out, inG, kerG, inShape, outShape, kernel, strides);
    }
    else
    {
        if (padding == PaddingType::kVALID)
            conv2d_grad_nchw<PaddingType::kVALID>(
                in, ker, out, inG, kerG, inShape, outShape, kernel, strides);
        else  // padding == PaddingType::kSAME
            conv2d_grad_nchw<PaddingType::kSAME>(
                in, ker, out, inG, kerG, inShape, outShape, kernel, strides);
    }
}

}  // namespace

Conv2DLayer::Conv2DLayer(ID id, const Tensor::SPtr& t,
                         const Tensor::SPtr& kernel,
                         const std::vector<int>& strides, PaddingType padding,
                         DataFormat dataFormat)
    : DifferentiableLayer(
          id, {t, kernel},
          {createOutput(t, kernel, strides, padding, dataFormat)}),
      mStrides(strides),
      mPadding(padding),
      mDataFormat(dataFormat),
      mGpuParams(t->getType(), 11)
{
    assert(t->getType() == kernel->getType());
}

Layer::TensorMap Conv2DLayer::gradients(Tensor::SPtr out, Tensor::SPtr outGrad)
{
    assert(mOutputs[0] == out);

    Tensor::SPtr in = mInputs[0].lock();
    Tensor::SPtr k = mInputs[1].lock();

    Layer::SPtr layer = createLayer<Conv2DGradientLayer>(
        in, k, out, outGrad, mStrides, mPadding, mDataFormat);

    return {{in, layer->getOutputs()[0]}, {k, layer->getOutputs()[1]}};
}

void Conv2DLayer::execute(const std::vector<float*>& inputs,
                          const std::vector<float*>& outputs,
                          const InputDict& /*inputDict*/)
{
    float* x = inputs[0];
    float* ker = inputs[1];
    float* y = outputs[0];

    std::vector<int> inShape = getInputs()[0]->getShape();
    std::vector<int> kerShape = getInputs()[1]->getShape();
    std::vector<int> outShape = getOutputs()[0]->getShape();

    if (mInputs[0].lock()->getType() == MemoryType::kHOST_MEMORY)
        runConv2DHost(x, ker, y, inShape, outShape, kerShape, mStrides,
                      mPadding, mDataFormat);
#ifdef CUDA_AVAILABLE
    else
    {
        /* std::vector<int> outShape = mOutputs[0]->getShape(); */
        /* size_t size = outShape[0] * outShape[1] * outShape[2] * outShape[3]; */
        /* cuda::runConv2DDevice(x, ker, y, size, mGpuParams.getValues(), */
        /*                       mPadding); */
    }
#endif
}

void Conv2DLayer::initialize()
{
    Tensor::SPtr inTensor = mInputs[0].lock();
    Tensor::SPtr kerTensor = mInputs[1].lock();

    std::vector<int> inShape = inTensor->getShape();
    std::vector<int> kShape = kerTensor->getShape();
    std::vector<int> outShape = mOutputs[0]->getShape();

    if (inTensor->getType() == MemoryType::kHOST_MEMORY)
    {
    }
#ifdef CUDA_AVAILABLE
    else
    {
    }
#endif
}

Conv2DLayer::~Conv2DLayer() = default;

Conv2DGradientLayer::Conv2DGradientLayer(
    ID id, const Tensor::SPtr& t, const Tensor::SPtr& k,
    const Tensor::SPtr& out, const Tensor::SPtr& outG, std::vector<int> strides,
    PaddingType padding, DataFormat dataFormat)
    : Layer(id, {t, k, out, outG},
            {createTensor("", t->getShape(), t->getType()),
             createTensor("", k->getShape(), k->getType())}),
      mStrides(std::move(strides)),
      mPadding(padding),
      mDataFormat(dataFormat),
      mGpuParams(t->getType(), 11)
{
}

void Conv2DGradientLayer::execute(const std::vector<float*>& inputs,
                                  const std::vector<float*>& outputs,
                                  const InputDict& /*inputDict*/)
{
    float* x = inputs[0];
    float* ker = inputs[1];
    float* yG = inputs[3];
    float* xG = outputs[0];
    float* kerG = outputs[1];

    std::vector<int> inShape = getInputs()[0]->getShape();
    std::vector<int> kerShape = getInputs()[1]->getShape();
    std::vector<int> outShape = getInputs()[2]->getShape();

    if (mInputs[0].lock()->getType() == MemoryType::kHOST_MEMORY)
        runConv2DGradientHost(x, ker, yG, xG, kerG, inShape, outShape, kerShape,
                              mStrides, mPadding, mDataFormat);
#ifdef CUDA_AVAILABLE
    else
    {
        /* std::vector<int> inShape = mInputs[0].lock()->getShape(); */
        /* std::vector<int> kerShape = mInputs[1].lock()->getShape(); */
        /* size_t inSize = inShape[0] * inShape[1] * inShape[2] * inShape[3]; */
        /* size_t kerSize = kerShape[0] * kerShape[1] * kerShape[2] * kerShape[3]; */
        /* cuda::runConv2DGradientDevice(x, ker, yG, xG, kerG, inSize, kerSize, */
        /*                               mGpuParams.getValues(), mPadding); */
    }
#endif
}

void Conv2DGradientLayer::initialize()
{
    Tensor::SPtr inTensor = mInputs[0].lock();
    Tensor::SPtr kerTensor = mInputs[1].lock();
    Tensor::SPtr outTensor = mInputs[2].lock();

    std::vector<int> inShape = inTensor->getShape();
    std::vector<int> kShape = kerTensor->getShape();
    std::vector<int> outShape = outTensor->getShape();

    if (inTensor->getType() == MemoryType::kHOST_MEMORY)
    {
    }
#ifdef CUDA_AVAILABLE
    else
    {
        mGpuParams.allocate();
    }
#endif
}

Conv2DGradientLayer::~Conv2DGradientLayer() = default;

}  // namespace layers

Tensor::SPtr convolution2D(const Tensor::SPtr& t, const Tensor::SPtr& kernel,
                           const std::vector<int>& strides,
                           layers::PaddingType padding,
                           layers::DataFormat dataFormat)
{
    TensorShape xShape = t->getShape();
    TensorShape kShape = kernel->getShape();

    if (xShape.size() != 4)
        throw std::runtime_error("conv2D: wrong input shape");
    if (kShape.size() != 4)
        throw std::runtime_error("conv2D: wrong kernel shape");
    if (strides.empty() || strides.size() > 2)
        throw std::runtime_error("conv2D: wrong strides");

    if (dataFormat == layers::DataFormat::kNHWC)
    {
        if (xShape[3] != kShape[2])
            throw std::runtime_error("conv2D: kernel doesn\'t match tensor");
    }
    else
    {
        if (xShape[1] != kShape[2])
            throw std::runtime_error("conv2D: kernel doesn\'t match tensor");
    }

    for (int d : strides)
        if (d <= 0)
            throw std::runtime_error("conv2D: stride dims must be positive");

    Layer::SPtr layer = createLayer<layers::Conv2DLayer>(t, kernel, strides,
                                                         padding, dataFormat);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr conv2D(const ITensorPtr& tensor, const ITensorPtr& kernel,
                  const std::vector<int>& strides, const std::string& padding,
                  const std::string& dataFormat)
{
    core::layers::PaddingType pad = core::layers::str2padding(padding);
    core::layers::DataFormat format = core::layers::str2format(dataFormat);

    core::Tensor::SPtr t = core::castITensorPtr(tensor)->get();
    core::Tensor::SPtr k = core::castITensorPtr(kernel)->get();
    return core::makeAbstractTensor(
        core::convolution2D(t, k, strides, pad, format));
}

}  // namespace graphdl
