#include "reduceSum.h"
#include <assert.h>
#include "dll_ops.h"
#include "graph.h"

namespace dll
{
namespace core
{
namespace layers
{
ReduceSumOper::ReduceSumOper(Tensor::SPtr tensor)
    : GradientOper({tensor}, {createTensor("", {})})
{
}

void ReduceSumOper::executeOper(const InputDict& inputs)
{
    Tensor::SPtr in = mInputs[0].lock();
    in->exec(inputs);

    Memory input = in->getMemory();
    Memory output = mOutputs[0]->getMemory();

    output[0] = 0.;
    for (std::size_t pos = 0; pos < input.count(); ++pos)
        output[0] += input[pos];
}

GradientOper::TensorMap ReduceSumOper::gradients(Tensor::SPtr out,
                                                 Tensor::SPtr outGrad)
{
    assert(out == mOutputs[0]);

    Tensor::SPtr in = getInputs()[0];
    Oper::SPtr oper = std::make_shared<ReduceSumGradientOper>(in, out, outGrad);
    getDefaultGraph()->insertOperation(oper);

    return {{in, oper->getOutputs()[0]}};
}

ReduceSumGradientOper::ReduceSumGradientOper(Tensor::SPtr in, Tensor::SPtr out,
                                             Tensor::SPtr outGrad)
    : Oper({in, out, outGrad}, {createTensor("", in->shape())})
{
}

void ReduceSumGradientOper::executeOper(const InputDict& inputs)
{
    Tensor::SPtr outputGrad = mInputs[2].lock();
    outputGrad->exec(inputs);

    Memory outGrad = outputGrad->getMemory();
    Memory inGrad = mOutputs[0]->getMemory();

    for (std::size_t pos = 0; pos < inGrad.count(); ++pos)
        inGrad[pos] = outGrad[0];
}

}  // namespace layers

Tensor::SPtr reduceSum(Tensor::SPtr t)
{
    Oper::SPtr oper = std::make_shared<layers::ReduceSumOper>(t);
    core::getDefaultGraph()->insertOperation(oper);
    return oper->getOutputs()[0];
}

}  // namespace core

ITensorSPtr reduceSum(ITensorSPtr t)
{
    core::Tensor::SPtr tensor = std::static_pointer_cast<core::Tensor>(t);
    return ITensorSPtr(core::reduceSum(tensor));
}

}  // namespace dll
