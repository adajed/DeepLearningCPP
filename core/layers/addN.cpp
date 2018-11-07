#include "addN.h"

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
std::vector<Tensor::SPtr> createGradientInputs(std::vector<Tensor::SPtr> ins,
                                               const Tensor::SPtr& out,
                                               const Tensor::SPtr& outGrad)
{
    ins.push_back(out);
    ins.push_back(outGrad);
    return ins;
}

std::vector<Tensor::SPtr> createGradientOutputs(std::vector<Tensor::SPtr> ins)
{
    std::vector<Tensor::SPtr> outs;
    outs.reserve(ins.size());
    for (const Tensor::SPtr& i : ins)
        outs.push_back(createTensor("", i->getShape()));
    return outs;
}

}  // namespace

AddNLayer::AddNLayer(ID id, std::vector<Tensor::SPtr> tensors)
    : DifferentiableLayer(id, tensors,
                          {createTensor("", tensors[0]->getShape())})
{
}

void AddNLayer::execute(const InputDict& inputs)
{
    std::vector<Tensor::SPtr> ins = getInputs();
    std::vector<Memory> inMemory;
    inMemory.reserve(ins.size());
    for (const Tensor::SPtr& in : ins)
    {
        in->eval(inputs);
        inMemory.push_back(in->getMemory());
    }
    Memory outMemory = mOutputs[0]->getMemory();

    for (std::size_t pos = 0; pos < outMemory.getCount(); ++pos)
    {
        outMemory[pos] = 0.;
        for (auto& i : inMemory) outMemory[pos] += i[pos];
    }
}

Layer::TensorMap AddNLayer::gradients(Tensor::SPtr out, Tensor::SPtr outGrad)
{
    assert(out == mOutputs[0]);

    std::vector<Tensor::SPtr> inputs = getInputs();
    Layer::SPtr layer = createLayer<AddNGradientLayer>(inputs, out, outGrad);

    std::vector<Tensor::SPtr> grads = layer->getOutputs();
    Layer::TensorMap gradMap;
    for (unsigned i = 0; i < inputs.size(); ++i)
        gradMap.insert({inputs[i], grads[i]});
    return gradMap;
}

AddNGradientLayer::AddNGradientLayer(ID id,
                                     const std::vector<Tensor::SPtr>& ins,
                                     const Tensor::SPtr& out,
                                     const Tensor::SPtr& outGrad)
    : Layer(id, createGradientInputs(ins, out, outGrad),
            createGradientOutputs(ins))
{
}

void AddNGradientLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr outputGrad = mInputs.back().lock();
    outputGrad->eval(inputs);

    Memory outG = outputGrad->getMemory();
    std::vector<Memory> inG;
    for (auto& mOutput : mOutputs) inG.push_back(mOutput->getMemory());

    for (std::size_t pos = 0; pos < outG.getCount(); ++pos)
    {
        for (auto& i : inG) i[pos] = outG[pos];
    }
}

}  // namespace layers

Tensor::SPtr addN(std::vector<Tensor::SPtr> tensors)
{
    if (tensors.empty())
        throw std::runtime_error("List of input tensors cannot be empty");
    for (unsigned i = 1; i < tensors.size(); ++i)
    {
        if (tensors[0]->getShape() != tensors[i]->getShape())
            throw std::runtime_error("Shapes of inputs tensors don\'t match");
    }

    Layer::SPtr layer = createLayer<layers::AddNLayer>(tensors);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr addN(std::vector<ITensorPtr> tensors)
{
    std::vector<core::Tensor::SPtr> inputs;
    inputs.reserve(tensors.size());
    for (const ITensorPtr& t : tensors)
        inputs.push_back(core::castITensorPtr(t)->get());

    return makeAbstractTensor(core::addN(inputs));
}

}  // namespace graphdl
