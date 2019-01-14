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
                                 PaddingType padding)
{
    TensorShape shape = t->getShape();

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

    return createTensor("", shape, t->getType());
}

void runPooling2DHost(const float* x, float* y, const std::vector<int>& shape,
                      const std::vector<int>& kernel,
                      const std::vector<int>& strides, PoolingType pooling,
                      PaddingType padding)
{
    if (pooling == PoolingType::kMAX)
    {
        if (padding == PaddingType::kVALID)
            pool<PoolingType::kMAX, PaddingType::kVALID>(x, y, shape, kernel,
                                                         strides);
        else  // padding == PaddingType::kSAME
            pool<PoolingType::kMAX, PaddingType::kSAME>(x, y, shape, kernel,
                                                        strides);
    }
    else  // pooling == PoolingType::kAVERAGE
    {
        if (padding == PaddingType::kVALID)
            pool<PoolingType::kAVERAGE, PaddingType::kVALID>(x, y, shape,
                                                             kernel, strides);
        else  // padding == PaddingType::kSAME
            pool<PoolingType::kAVERAGE, PaddingType::kSAME>(x, y, shape, kernel,
                                                            strides);
    }
}

void runPooling2DGradientHost(const float* in, const float* out,
                              const float* outG, float* inG,
                              const std::vector<int>& shape,
                              const std::vector<int>& kernel,
                              const std::vector<int>& strides,
                              PoolingType pooling, PaddingType padding)
{
    if (pooling == PoolingType::kMAX)
    {
        if (padding == PaddingType::kVALID)
            poolGradient<PoolingType::kMAX, PaddingType::kVALID>(
                in, out, outG, inG, shape, kernel, strides);
        else  // padding == PaddingType::kSAME
            poolGradient<PoolingType::kMAX, PaddingType::kSAME>(
                in, out, outG, inG, shape, kernel, strides);
    }
    else  // pooling == PoolingType::kAVERAGE
    {
        if (padding == PaddingType::kVALID)
            poolGradient<PoolingType::kAVERAGE, PaddingType::kVALID>(
                in, out, outG, inG, shape, kernel, strides);
        else  // padding == PaddingType::kSAME
            poolGradient<PoolingType::kAVERAGE, PaddingType::kSAME>(
                in, out, outG, inG, shape, kernel, strides);
    }
}

}  // namespace

Pooling2DLayer::Pooling2DLayer(ID id, const Tensor::SPtr& t,
                               PoolingType pooling,
                               const std::vector<int>& kernel,
                               const std::vector<int>& strides,
                               PaddingType padding)
    : DifferentiableLayer(id, {t},
                          {createPoolingOutput(t, kernel, strides, padding)}),
      mPooling(pooling),
      mKernelWindow(kernel),
      mStrides(strides),
      mPadding(padding)
{
}

Layer::TensorMap Pooling2DLayer::gradients(Tensor::SPtr out,
                                           Tensor::SPtr outGrad)
{
    assert(out == mOutputs[0]);

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
    float* out = mOutputs[0]->getMemory().getValues();

    TensorShape tShape = inTensor->getShape();
    std::vector<int> shape(tShape.begin(), tShape.end());

    if (inTensor->getType() == MemoryType::kHOST_MEMORY)
        runPooling2DHost(in, out, shape, mKernelWindow, mStrides, mPooling,
                         mPadding);
    else  // inTensor->getType() == MemoryType::kDEVICE_MEMORY
        cuda::runPool2DDevice(in, out, shape.data(), mKernelWindow.data(),
                              mStrides.data(), mPooling, mPadding);
}

Pooling2DGradientLayer::Pooling2DGradientLayer(ID id, const Tensor::SPtr& t,
                                               const Tensor::SPtr& out,
                                               const Tensor::SPtr& outGrad,
                                               PoolingType pooling,
                                               const std::vector<int>& kernel,
                                               const std::vector<int>& strides,
                                               PaddingType padding)
    : Layer(id, {t, out, outGrad},
            {createTensor("", t->getShape(), t->getType())}),
      mPooling(pooling),
      mKernelWindow(kernel),
      mStrides(strides),
      mPadding(padding)
{
}

void Pooling2DGradientLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr inTensor = mInputs[0].lock();
    Tensor::SPtr outTensor = mInputs[1].lock();
    Tensor::SPtr outGradTensor = mInputs[2].lock();
    inTensor->eval(inputs);
    outTensor->eval(inputs);
    outGradTensor->eval(inputs);

    float* in = inTensor->getMemory().getValues();
    float* out = outTensor->getMemory().getValues();
    float* outG = outGradTensor->getMemory().getValues();
    float* inG = mOutputs[0]->getMemory().getValues();

    TensorShape tShape = inTensor->getShape();
    std::vector<int> shape(tShape.begin(), tShape.end());

    if (outGradTensor->getType() == MemoryType::kHOST_MEMORY)
        runPooling2DGradientHost(in, out, outG, inG, shape, mKernelWindow,
                                 mStrides, mPooling, mPadding);
    else  // outGradTensor->getType() == MemoryType::kDEVICE_MEMORY
        cuda::runPool2DGradientDevice(in, out, outG, inG, shape.data(),
                                      mKernelWindow.data(), mStrides.data(),
                                      mPooling, mPadding);
}

}  // namespace layers

Tensor::SPtr pooling2D(const Tensor::SPtr& t, layers::PoolingType pooling,
                       const std::vector<int>& kernel,
                       const std::vector<int>& strides,
                       layers::PaddingType padding)
{
    if (t->getShape().size() != 4)
        throw std::runtime_error("pool2D: wrong input shape");
    if (kernel.empty() || kernel.size() > 2)
        throw std::runtime_error("pool2D: wrong kernel");
    if (strides.empty() || strides.size() > 2)
        throw std::runtime_error("pool2D: wrong strides");

    std::vector<int> kernel2 = kernel;
    if (kernel2.size() == 1) kernel2.push_back(kernel2[0]);
    std::vector<int> strides2 = strides;
    if (strides2.size() == 1) strides2.push_back(strides2[0]);

    Layer::SPtr layer = createLayer<layers::Pooling2DLayer>(t, pooling, kernel2,
                                                            strides2, padding);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr maxPool2D(const ITensorPtr& tensor, const std::vector<int>& kernel,
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
    return core::makeAbstractTensor(core::pooling2D(
        t, core::layers::PoolingType::kMAX, kernel, strides, p));
}

ITensorPtr avgPool2D(const ITensorPtr& tensor, const std::vector<int>& kernel,
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
            R"(avgPool2D: wrong padding, should be one of: "SAME", "VALID")");

    core::Tensor::SPtr t = core::castITensorPtr(tensor)->get();
    return core::makeAbstractTensor(core::pooling2D(
        t, core::layers::PoolingType::kAVERAGE, kernel, strides, p));
}

}  // namespace graphdl
