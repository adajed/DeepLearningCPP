#include "reduceSum.h"
#include <assert.h>
#include "abstractTensor.h"
#include "graph.h"
#include "graphdl_ops.h"

namespace graphdl
{
namespace core
{
namespace layers
{
ReduceSumLayer::ReduceSumLayer(ID id, Tensor::SPtr tensor)
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

    output[0] = 0.;
    for (std::size_t pos = 0; pos < size; ++pos) output[0] += input[pos];
}

Layer::TensorMap ReduceSumLayer::gradients(Tensor::SPtr out,
                                           Tensor::SPtr outGrad)
{
    assert(out == mOutputs[0]);

    Tensor::SPtr in = getInputs()[0];
    Layer::SPtr layer = createLayer<ReduceSumGradientLayer>(in, out, outGrad);
    return {{in, layer->getOutputs()[0]}};
}

ReduceSumGradientLayer::ReduceSumGradientLayer(ID id, Tensor::SPtr in,
                                               Tensor::SPtr out,
                                               Tensor::SPtr outGrad)
    : Layer(id, {in, out, outGrad},
            {createTensor("", in->getShape(), outGrad->getType())})
{
}

void ReduceSumGradientLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr outputGrad = mInputs[2].lock();
    outputGrad->eval(inputs);

    float* outGrad = outputGrad->getMemory().getValues();
    float* inGrad = mOutputs[0]->getMemory().getValues();
    std::size_t size = mOutputs[0]->getMemory().getCount();

    for (std::size_t pos = 0; pos < size; ++pos) inGrad[pos] = outGrad[0];
}

}  // namespace layers

Tensor::SPtr reduceSum(Tensor::SPtr t)
{
    Layer::SPtr layer = createLayer<layers::ReduceSumLayer>(t);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr reduceSum(ITensorPtr t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::reduceSum(tensor->get()));
}

}  // namespace graphdl
