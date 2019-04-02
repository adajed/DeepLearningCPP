#include "pooling.h"

#include "abstractTensor.h"
#include "graph.h"
#include "graphdl.h"
#include "graphdl_ops.h"
#include "pooling_host.h"

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
Tensor::SPtr createPoolingOutput(const Tensor::SPtr& t,
                                 const std::vector<int>& kernel,
                                 const std::vector<int>& strides,
                                 PaddingType padding, DataFormat dataFormat)
{
    TensorShape shape = t->getShape();

    if (dataFormat == DataFormat::kNHWC)
    {
        if (padding == PaddingType::kVALID)
        {
            shape[1] = ceil(shape[1] - kernel[0] + 1, strides[0]);
            shape[2] = ceil(shape[2] - kernel[1] + 1, strides[1]);
        }
        else  // padding == PaddingType::kSAME
        {
            shape[1] = ceil(shape[1], strides[0]);
            shape[2] = ceil(shape[2], strides[1]);
        }
    }
    else  // dataFormat == DataFormat::kNCHW
    {
        if (padding == PaddingType::kVALID)
        {
            shape[2] = ceil(shape[2] - kernel[0] + 1, strides[0]);
            shape[3] = ceil(shape[3] - kernel[1] + 1, strides[1]);
        }
        else  // padding == PaddingType::kSAME
        {
            shape[2] = ceil(shape[2], strides[0]);
            shape[3] = ceil(shape[3], strides[1]);
        }
    }

    return createTensor("", shape, t->getType());
}

void runPooling2DHost(const float* x, float* y, const std::vector<int>& inShape,
                      const std::vector<int>& outShape,
                      const std::vector<int>& kernel,
                      const std::vector<int>& strides, PoolingType pooling,
                      PaddingType padding, DataFormat dataFormat)
{
    if (dataFormat == DataFormat::kNHWC)
    {
        if (pooling == PoolingType::kMAX)
        {
            if (padding == PaddingType::kVALID)
                pool_max_nhwc<PaddingType::kVALID>(x, y, inShape, outShape,
                                                   kernel, strides);
            else  // padding == PaddingType::kSAME
                pool_max_nhwc<PaddingType::kSAME>(x, y, inShape, outShape,
                                                  kernel, strides);
        }
        else  // pooling == PoolingType::kAVERAGE
        {
            if (padding == PaddingType::kVALID)
                pool_avg_nhwc<PaddingType::kVALID>(x, y, inShape, outShape,
                                                   kernel, strides);
            else  // padding == PaddingType::kSAME
                pool_avg_nhwc<PaddingType::kSAME>(x, y, inShape, outShape,
                                                  kernel, strides);
        }
    }
    else  // dataFormat == DataFormat::kNCHW
    {
        if (pooling == PoolingType::kMAX)
        {
            if (padding == PaddingType::kVALID)
                pool_max_nchw<PaddingType::kVALID>(x, y, inShape, outShape,
                                                   kernel, strides);
            else  // padding == PaddingType::kSAME
                pool_max_nchw<PaddingType::kSAME>(x, y, inShape, outShape,
                                                  kernel, strides);
        }
        else  // pooling == PoolingType::kAVERAGE
        {
            if (padding == PaddingType::kVALID)
                pool_avg_nchw<PaddingType::kVALID>(x, y, inShape, outShape,
                                                   kernel, strides);
            else  // padding == PaddingType::kSAME
                pool_avg_nchw<PaddingType::kSAME>(x, y, inShape, outShape,
                                                  kernel, strides);
        }
    }
}

void runPooling2DGradientHost(const float* x, const float* y, const float* yG,
                              float* xG, const std::vector<int>& inShape,
                              const std::vector<int>& outShape,
                              const std::vector<int>& kernel,
                              const std::vector<int>& strides,
                              PoolingType pooling, PaddingType padding,
                              DataFormat dataFormat)
{
    if (dataFormat == DataFormat::kNHWC)
    {
        if (pooling == PoolingType::kMAX)
        {
            if (padding == PaddingType::kVALID)
                pool_grad_max_nhwc<PaddingType::kVALID>(
                    x, y, yG, xG, inShape, outShape, kernel, strides);
            else  // padding == PaddingType::kSAME
                pool_grad_max_nhwc<PaddingType::kSAME>(
                    x, y, yG, xG, inShape, outShape, kernel, strides);
        }
        else  // pooling == PoolingType::kAVERAGE
        {
            if (padding == PaddingType::kVALID)
                pool_grad_avg_nhwc<PaddingType::kVALID>(
                    x, y, yG, xG, inShape, outShape, kernel, strides);
            else  // padding == PaddingType::kSAME
                pool_grad_avg_nhwc<PaddingType::kSAME>(
                    x, y, yG, xG, inShape, outShape, kernel, strides);
        }
    }
    else  // dataFormat == DataFormat::kNCHW
    {
        if (pooling == PoolingType::kMAX)
        {
            if (padding == PaddingType::kVALID)
                pool_grad_max_nchw<PaddingType::kVALID>(
                    x, y, yG, xG, inShape, outShape, kernel, strides);
            else  // padding == PaddingType::kSAME
                pool_grad_max_nchw<PaddingType::kSAME>(
                    x, y, yG, xG, inShape, outShape, kernel, strides);
        }
        else  // pooling == PoolingType::kAVERAGE
        {
            if (padding == PaddingType::kVALID)
                pool_grad_avg_nchw<PaddingType::kVALID>(
                    x, y, yG, xG, inShape, outShape, kernel, strides);
            else  // padding == PaddingType::kSAME
                pool_grad_avg_nchw<PaddingType::kSAME>(
                    x, y, yG, xG, inShape, outShape, kernel, strides);
        }
    }
}

}  // namespace

PaddingType str2padding(const std::string& s)
{
    if (s == "SAME" || s == "same") return PaddingType::kSAME;
    if (s == "VALID" || s == "valid") return PaddingType::kVALID;

    throw std::runtime_error(
        R"(Wrong padding type, must be one of: "SAME", "VALID".)");
}

DataFormat str2format(const std::string& s)
{
    if (s == "NHWC" || s == "nhwc") return DataFormat::kNHWC;
    if (s == "NCHW" || s == "nchw") return DataFormat::kNCHW;

    throw std::runtime_error(
        R"(Wrong data format type, must be one of: "NHWC", "NCHW".)");
}

Pooling2DLayer::Pooling2DLayer(ID id, const Tensor::SPtr& t,
                               PoolingType pooling,
                               const std::vector<int>& kernel,
                               const std::vector<int>& strides,
                               PaddingType padding, DataFormat dataFormat)
    : DifferentiableLayer(
          id, {t},
          {createPoolingOutput(t, kernel, strides, padding, dataFormat)}),
      mPooling(pooling),
      mKernelWindow(kernel),
      mStrides(strides),
      mPadding(padding),
      mDataFormat(dataFormat),
      mGpuParams(t->getType(), 11)
{
}

Layer::TensorMap Pooling2DLayer::gradients(Tensor::SPtr out,
                                           Tensor::SPtr outGrad)
{
    assert(out == mOutputs[0]);

    Tensor::SPtr input = mInputs[0].lock();
    Layer::SPtr layer = createLayer<Pooling2DGradientLayer>(
        input, out, outGrad, mPooling, mKernelWindow, mStrides, mPadding,
        mDataFormat);
    return {{input, layer->getOutputs()[0]}};
}

void Pooling2DLayer::execute(const std::vector<float*>& inputs,
                             const std::vector<float*>& outputs,
                             const InputDict& /*inputDict*/)
{
    float* x = inputs[0];
    float* y = outputs[0];

    Tensor::SPtr tX = getInputs()[0];
    std::vector<int> inShape = tX->getShape();
    std::vector<int> outShape = getOutputs()[0]->getShape();

    if (tX->getType() == MemoryType::kHOST_MEMORY)
        runPooling2DHost(x, y, inShape, outShape, mKernelWindow, mStrides,
                         mPooling, mPadding, mDataFormat);
#ifdef CUDA_AVAILABLE
    else  // inTensor->getType() == MemoryType::kDEVICE_MEMORY
    {
        std::vector<int> shapeY = mOutputs[0]->getShape();
        size_t size = shapeY[0] * shapeY[1] * shapeY[2] * shapeY[3];
        cuda::runPool2DDevice(x, y, mGpuParams.getValues(), size, mPooling,
                              mPadding);
    }
#endif
}

void Pooling2DLayer::initialize()
{
    Tensor::SPtr tX = getInputs()[0];
    std::vector<int> shapeX = tX->getShape();
    std::vector<int> shapeY = mOutputs[0]->getShape();

    if (tX->getType() == MemoryType::kHOST_MEMORY)
    {
    }
#ifdef CUDA_AVAILABLE
    else
    {
        mGpuParams.allocate();
        cuda::initializePoolGpuParams(mGpuParams.getValues(), shapeX.data(),
                                      mKernelWindow.data(), mStrides.data(),
                                      shapeY.data());
    }
#endif
}

Pooling2DLayer::~Pooling2DLayer()
{
    mGpuParams.free();
}

Pooling2DGradientLayer::Pooling2DGradientLayer(
    ID id, const Tensor::SPtr& t, const Tensor::SPtr& out,
    const Tensor::SPtr& outGrad, PoolingType pooling, std::vector<int> kernel,
    std::vector<int> strides, PaddingType padding, DataFormat dataFormat)
    : Layer(id, {t, out, outGrad},
            {createTensor("", t->getShape(), t->getType())}),
      mPooling(pooling),
      mKernelWindow(std::move(kernel)),
      mStrides(std::move(strides)),
      mPadding(padding),
      mDataFormat(dataFormat),
      mGpuParams(t->getType(), 11)
{
}

void Pooling2DGradientLayer::execute(const std::vector<float*>& inputs,
                                     const std::vector<float*>& outputs,
                                     const InputDict& /*inputDict*/)
{
    float* x = inputs[0];
    float* y = inputs[1];
    float* yGrad = inputs[2];
    float* xGrad = outputs[0];

    Tensor::SPtr tX = getInputs()[0];
    std::vector<int> inShape = tX->getShape();
    std::vector<int> outShape = getInputs()[1]->getShape();

    if (tX->getType() == MemoryType::kHOST_MEMORY)
        runPooling2DGradientHost(x, y, yGrad, xGrad, inShape, outShape,
                                 mKernelWindow, mStrides, mPooling, mPadding,
                                 mDataFormat);
#ifdef CUDA_AVAILABLE
    else  // outGradTensor->getType() == MemoryType::kDEVICE_MEMORY
    {
        size_t size = shapeX[0] * shapeX[1] * shapeX[2] * shapeX[3];
        cuda::runPool2DGradientDevice(x, y, yGrad, xGrad,
                                      mGpuParams.getValues(), size, mPooling,
                                      mPadding);
    }
#endif
}

void Pooling2DGradientLayer::initialize()
{
    Tensor::SPtr inTensor = mInputs[0].lock();
    Tensor::SPtr outTensor = mInputs[1].lock();
    std::vector<int> inShape = inTensor->getShape();
    std::vector<int> outShape = outTensor->getShape();

    if (inTensor->getType() == MemoryType::kHOST_MEMORY)
    {
    }
#ifdef CUDA_AVAILABLE
    else
    {
        mGpuParams.allocate();
        cuda::initializePoolGpuParams(mGpuParams.getValues(), inShape.data(),
                                      mKernelWindow.data(), mStrides.data(),
                                      outShape.data());
    }
#endif
}

Pooling2DGradientLayer::~Pooling2DGradientLayer()
{
    mGpuParams.free();
}

}  // namespace layers

Tensor::SPtr pooling2D(const Tensor::SPtr& t, layers::PoolingType pooling,
                       const std::vector<int>& kernel,
                       const std::vector<int>& strides,
                       layers::PaddingType padding,
                       layers::DataFormat dataFormat)
{
    if (t->getShape().size() != 4)
        throw std::runtime_error("pool2D: wrong input shape");
    if (kernel.empty() || kernel.size() > 2)
        throw std::runtime_error("pool2D: wrong kernel");
    if (strides.empty() || strides.size() > 2)
        throw std::runtime_error("pool2D: wrong strides");

    for (int d : kernel)
        if (d <= 0)
            throw std::runtime_error("pool2D: kernel dims must be positive");
    for (int d : strides)
        if (d <= 0)
            throw std::runtime_error("pool2D: stride dims must be positive");

    std::vector<int> kernel2 = kernel;
    if (kernel2.size() == 1) kernel2.push_back(kernel2[0]);
    std::vector<int> strides2 = strides;
    if (strides2.size() == 1) strides2.push_back(strides2[0]);

    Layer::SPtr layer = createLayer<layers::Pooling2DLayer>(
        t, pooling, kernel2, strides2, padding, dataFormat);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr maxPool2D(const ITensorPtr& tensor, const std::vector<int>& kernel,
                     const std::vector<int>& strides,
                     const std::string& padding, const std::string& format)
{
    core::layers::PaddingType pad = core::layers::str2padding(padding);
    core::layers::DataFormat dataFormat = core::layers::str2format(format);

    core::Tensor::SPtr t = core::castITensorPtr(tensor)->get();
    return core::makeAbstractTensor(core::pooling2D(
        t, core::layers::PoolingType::kMAX, kernel, strides, pad, dataFormat));
}

ITensorPtr avgPool2D(const ITensorPtr& tensor, const std::vector<int>& kernel,
                     const std::vector<int>& strides,
                     const std::string& padding, const std::string& format)
{
    core::layers::PaddingType pad = core::layers::str2padding(padding);
    core::layers::DataFormat dataFormat = core::layers::str2format(format);

    core::Tensor::SPtr t = core::castITensorPtr(tensor)->get();
    return core::makeAbstractTensor(
        core::pooling2D(t, core::layers::PoolingType::kAVERAGE, kernel, strides,
                        pad, dataFormat));
}

}  // namespace graphdl
