#include "addN.h"
#include "dll_ops.h"
#include "graph.h"

namespace dll
{
namespace core
{
namespace layers
{
std::vector<Tensor::SPtr> createOutput(std::vector<Tensor::SPtr> ins)
{
    assert(ins.size() > 0);
    for (unsigned i = 1; i < ins.size(); ++i)
        assert(ins[0]->shape() == ins[i]->shape());
    return {createTensor("", ins[0]->shape())};
}

AddNOper::AddNOper(std::vector<Tensor::SPtr> tensors)
    : GradientOper(tensors, createOutput(tensors))
{
}

void AddNOper::executeOper(const InputDict& inputs)
{
    std::vector<Tensor::SPtr> ins = getInputs();
    std::vector<Memory> inMemory;
    for (Tensor::SPtr in : ins)
    {
        in->exec(inputs);
        inMemory.push_back(in->getMemory());
    }
    Memory outMemory = mOutputs[0]->getMemory();

    for (std::size_t pos = 0; pos < outMemory.count(); ++pos)
    {
        outMemory[pos] = 0.;
        for (unsigned i = 0; i < inMemory.size(); ++i)
            outMemory[pos] += inMemory[i][pos];
    }
}

GradientOper::TensorMap AddNOper::gradients(Tensor::SPtr out,
                                            Tensor::SPtr outGrad)
{
    assert(out == mOutputs[0]);

    std::vector<Tensor::SPtr> inputs = getInputs();
    Oper::SPtr oper = std::make_shared<AddNGradientOper>(inputs, out, outGrad);
    core::getDefaultGraph()->insertOperation(oper);
    std::vector<Tensor::SPtr> grads = oper->getOutputs();

    GradientOper::TensorMap gradMap;
    for (unsigned i = 0; i < inputs.size(); ++i)
        gradMap.insert({inputs[i], grads[i]});
    return gradMap;
}

std::vector<Tensor::SPtr> createGradientInputs(std::vector<Tensor::SPtr> ins,
                                               Tensor::SPtr out,
                                               Tensor::SPtr outGrad)
{
    ins.push_back(out);
    ins.push_back(outGrad);
    return ins;
}

std::vector<Tensor::SPtr> createGradientOutputs(std::vector<Tensor::SPtr> ins)
{
    assert(ins.size() > 0);
    for (unsigned i = 1; i < ins.size(); ++i)
        assert(ins[0]->shape() == ins[i]->shape());

    std::vector<Tensor::SPtr> outs;
    for (Tensor::SPtr i : ins) outs.push_back(createTensor("", i->shape()));
    return outs;
}

AddNGradientOper::AddNGradientOper(std::vector<Tensor::SPtr> ins,
                                   Tensor::SPtr out, Tensor::SPtr outGrad)
    : Oper(createGradientInputs(ins, out, outGrad), createGradientOutputs(ins))
{
}

void AddNGradientOper::executeOper(const InputDict& inputs)
{
    Tensor::SPtr outputGrad = mInputs.back().lock();
    outputGrad->exec(inputs);

    Memory outG = outputGrad->getMemory();
    std::vector<Memory> inG;
    for (unsigned i = 0; i < mOutputs.size(); ++i)
        inG.push_back(mOutputs[i]->getMemory());

    for (std::size_t pos = 0; pos < outG.count(); ++pos)
    {
        for (unsigned i = 0; i < inG.size(); ++i) inG[i][pos] = outG[pos];
    }
}

}  // namespace layers

Tensor::SPtr addN(std::vector<Tensor::SPtr> tensors)
{
    if (tensors.size() == 0)
        throw std::invalid_argument("List of input tensors cannot be empty");
    for (unsigned i = 1; i < tensors.size(); ++i)
    {
        if (tensors[0]->shape() != tensors[i]->shape())
            throw std::invalid_argument(
                "Shapes of inputs tensors don\'t match");
    }

    Oper::SPtr oper = std::make_shared<layers::AddNOper>(tensors);
    core::getDefaultGraph()->insertOperation(oper);
    return oper->getOutputs()[0];
}

}  // namespace core

ITensorSPtr addN(std::vector<ITensorSPtr> tensors)
{
    std::vector<core::Tensor::SPtr> inputs;
    for (ITensorSPtr t : tensors)
        inputs.push_back(std::static_pointer_cast<core::Tensor>(t));

    return ITensorSPtr(core::addN(inputs));
}

}  // namespace dll
