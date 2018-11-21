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
void runReduceSumHost(std::size_t size, const float* x, float* y)
{
    y[0] = 0.;
    for (std::size_t pos = 0; pos < size; ++pos) y[0] += x[pos];
}

void runReduceSumGradientHost(std::size_t size, const float* yGrad,
                              float* xGrad)
{
    for (std::size_t pos = 0; pos < size; ++pos) xGrad[pos] = yGrad[0];
}

}  // namespace

ReduceSumLayer::ReduceSumLayer(ID id, const Tensor::SPtr& tensor)
    : DifferentiableLayer(id, {tensor},
                          {createTensor("", {}, tensor->getType())})
{
}

void ReduceSumLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr in = mInputs[0].lock();
    in->eval(inputs);

    float* input = in->getMemory().getValues();
    float* output = mOutputs[0]->getMemory().getValues();
    std::size_t size = in->getMemory().getCount();

    if (in->getType() == MemoryType::kHOST_MEMORY)
        runReduceSumHost(size, input, output);
#ifdef CUDA_AVAILABLE
    else
        cuda::runReduceSumDevice(input, size, output);
#endif
}

Layer::TensorMap ReduceSumLayer::gradients(Tensor::SPtr out,
                                           Tensor::SPtr outGrad)
{
    assert(out == mOutputs[0]);

    Tensor::SPtr in = getInputs()[0];
    Layer::SPtr layer = createLayer<ReduceSumGradientLayer>(in, out, outGrad);
    return {{in, layer->getOutputs()[0]}};
}

ReduceSumGradientLayer::ReduceSumGradientLayer(ID id, const Tensor::SPtr& in,
                                               Tensor::SPtr out,
                                               Tensor::SPtr outGrad)
    : Layer(id, {in, std::move(out), std::move(outGrad)},
            {createTensor("", in->getShape(), in->getType())})
{
}

void ReduceSumGradientLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr outputGrad = mInputs[2].lock();
    outputGrad->eval(inputs);

    float* outGrad = outputGrad->getMemory().getValues();
    float* inGrad = mOutputs[0]->getMemory().getValues();
    std::size_t size = mOutputs[0]->getMemory().getCount();

    if (outputGrad->getType() == MemoryType::kHOST_MEMORY)
        runReduceSumGradientHost(size, outGrad, inGrad);
#ifdef CUDA_AVAILABLE
    else
        cuda::runReduceSumGradientDevice(outGrad, size, inGrad);
#endif
}

}  // namespace layers

Tensor::SPtr reduceSum(Tensor::SPtr t)
{
    Layer::SPtr layer = createLayer<layers::ReduceSumLayer>(std::move(t));
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr reduceSum(const ITensorPtr& t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::reduceSum(tensor->get()));
}

}  // namespace graphdl
