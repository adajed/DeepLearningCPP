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
ReduceSumLayer::ReduceSumLayer(ID id, Tensor::SPtr tensor)
    : DifferentiableLayer(id, {std::move(tensor)}, {createTensor("", {})})
{
}

void ReduceSumLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr in = mInputs[0].lock();
    in->eval(inputs);

    Memory input = in->getMemory();
    Memory output = mOutputs[0]->getMemory();

    output[0] = 0.;
    for (std::size_t pos = 0; pos < input.getCount(); ++pos)
        output[0] += input[pos];
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
            {createTensor("", in->getShape())})
{
}

void ReduceSumGradientLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr outputGrad = mInputs[2].lock();
    outputGrad->eval(inputs);

    Memory outGrad = outputGrad->getMemory();
    Memory inGrad = mOutputs[0]->getMemory();

    for (std::size_t pos = 0; pos < inGrad.getCount(); ++pos)
        inGrad[pos] = outGrad[0];
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
