#include "reduceSum.h"

#include "abstractTensor.h"
#include "graph.h"
#include "graphdl_ops.h"

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
Tensor::SPtr createOutput(const Tensor::SPtr& t, int numAxes)
{
    std::vector<int> shape(t->getShape().size() - numAxes, 1);
    for (unsigned i = 0; i < shape.size(); ++i) shape[i] = t->getShape()[i];

    return createTensor("", shape, t->getType());
}

}  // namespace

void runReduceSumHost(const float* x, float* y, size_t outSize,
                      size_t reduceSize)
{
    for (size_t posY = 0; posY < outSize; ++posY)
    {
        y[posY] = 0.;
        for (size_t i = 0; i < reduceSize; ++i) y[posY] += x[i];
        x += reduceSize;
    }
}

void runReduceSumGradientHost(const float* yGrad, float* xGrad, size_t outSize,
                              size_t reduceSize)
{
    for (size_t posY = 0; posY < outSize; ++posY)
    {
        for (size_t i = 0; i < reduceSize; ++i) xGrad[i] = yGrad[posY];
        xGrad += reduceSize;
    }
}

ReduceSumLayer::ReduceSumLayer(ID id, const Tensor::SPtr& tensor, int numAxes)
    : DifferentiableLayer(id, {tensor}, {createOutput(tensor, numAxes)}),
      mNumAxes(numAxes)
{
}

void ReduceSumLayer::execute(const std::vector<float*>& inputs,
                             const std::vector<float*>& outputs,
                             const InputDict& /*inputDict*/)
{
    float* x = inputs[0];
    float* y = outputs[0];
    std::vector<int> shape = mInputs[0].lock()->getShape();
    size_t outSize = 1, reduceSize = 1;
    for (unsigned i = 0; i < shape.size() - mNumAxes; ++i) outSize *= shape[i];
    for (unsigned i = shape.size() - mNumAxes; i < shape.size(); ++i)
        reduceSize *= shape[i];

    if (mInputs[0].lock()->getType() == MemoryType::kHOST_MEMORY)
        runReduceSumHost(x, y, outSize, reduceSize);
#ifdef CUDA_AVAILABLE
    else
        cuda::runReduceSumDevice(x, y, outSize, reduceSize);
#endif
}

Layer::TensorMap ReduceSumLayer::gradients(Tensor::SPtr out,
                                           Tensor::SPtr outGrad)
{
    assert(out == mOutputs[0]);

    Tensor::SPtr in = getInputs()[0];
    Layer::SPtr layer =
        createLayer<ReduceSumGradientLayer>(in, mNumAxes, out, outGrad);
    return {{in, layer->getOutputs()[0]}};
}

ReduceSumGradientLayer::ReduceSumGradientLayer(ID id, const Tensor::SPtr& in,
                                               int numAxes, Tensor::SPtr out,
                                               Tensor::SPtr outGrad)
    : Layer(id, {in, std::move(out), std::move(outGrad)},
            {createTensor("", in->getShape(), in->getType())}),
      mNumAxes(numAxes)
{
}

void ReduceSumGradientLayer::execute(const std::vector<float*>& inputs,
                                     const std::vector<float*>& outputs,
                                     const InputDict& /*inputDict*/)
{
    float* yGrad = inputs[2];
    float* xGrad = outputs[0];
    std::vector<int> shape = mOutputs[0]->getShape();
    size_t outSize = 1, reduceSize = 1;
    for (unsigned i = 0; i < shape.size() - mNumAxes; ++i) outSize *= shape[i];
    for (unsigned i = shape.size() - mNumAxes; i < shape.size(); ++i)
        reduceSize *= shape[i];

    if (mInputs[0].lock()->getType() == MemoryType::kHOST_MEMORY)
        runReduceSumGradientHost(yGrad, xGrad, outSize, reduceSize);
#ifdef CUDA_AVAILABLE
    else
        cuda::runReduceSumGradientDevice(yGrad, xGrad, outSize, reduceSize);
#endif
}

}  // namespace layers

Tensor::SPtr reduceSum(Tensor::SPtr t, int numAxes)
{
    if (numAxes <= 0) numAxes = t->getShape().size();
    Layer::SPtr layer =
        createLayer<layers::ReduceSumLayer>(std::move(t), numAxes);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr reduceSum(const ITensorPtr& t, int numAxes)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::reduceSum(tensor->get(), numAxes));
}

}  // namespace graphdl
