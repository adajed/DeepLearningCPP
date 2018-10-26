#include "addOper.h"
#include "dll_errors.h"
#include "dll_ops.h"
#include "graph.h"

namespace dll
{
namespace core
{
namespace layers
{
AddGradientOper::AddGradientOper(Tensor::SPtr input1, Tensor::SPtr input2,
                                 Tensor::SPtr output, Tensor::SPtr outputGrad)
    : Oper({input1, input2, output, outputGrad}, createOutputs(input1, input2))
{
}

std::vector<Tensor::SPtr> AddGradientOper::createOutputs(Tensor::SPtr i1,
                                                         Tensor::SPtr i2)
{
    /* assert(i1->shape() == i2->shape() && */
    /*        i1->shape() == out->shape()); */
    Tensor::SPtr grad1 = std::make_shared<Tensor>("", i1->shape());
    Tensor::SPtr grad2 = std::make_shared<Tensor>("", i2->shape());
    return {grad1, grad2};
}

void AddGradientOper::executeOper(const InputDict& inputs)
{
    Tensor::SPtr outputGrad = mInputs[3].lock();
    outputGrad->exec(inputs);

    Memory outGrad = outputGrad->getMemory();
    Memory in1Grad = mOutputs[0]->getMemory();
    Memory in2Grad = mOutputs[1]->getMemory();

    for (std::size_t i = 0; i < outGrad.count(); ++i)
        in1Grad[i] = in2Grad[i] = outGrad[i];
}

GradientOper::TensorMap AddOper::gradients(Tensor::SPtr output,
                                           Tensor::SPtr outputGrad)
{
    assert(output == mOutputs[0]);

    std::vector<Tensor::SPtr> inputs = getInputs();

    Oper::SPtr gradOper = Oper::SPtr(std::make_shared<AddGradientOper>(
        inputs[0], inputs[1], output, outputGrad));
    getDefaultGraph()->insertOperation(gradOper);
    std::vector<Tensor::SPtr> grads = gradOper->getOutputs();

    return {{inputs[0], grads[0]}, {inputs[1], grads[1]}};
}

}  // namespace layers

Tensor::SPtr add(Tensor::SPtr t1, Tensor::SPtr t2)
{
    if (t1->shape() != t2->shape()) throw errors::NotMatchingShapesError();
    Oper::SPtr oper = std::make_shared<layers::AddOper>(t1, t2);
    getDefaultGraph()->insertOperation(oper);
    return oper->getOutputs()[0];
}

Tensor::SPtr operator+(Tensor::SPtr t1, Tensor::SPtr t2) { return add(t1, t2); }

}  // namespace core

ITensorSPtr add(ITensorSPtr t1, ITensorSPtr t2)
{
    core::Tensor::SPtr tensor1 = std::static_pointer_cast<core::Tensor>(t1);
    core::Tensor::SPtr tensor2 = std::static_pointer_cast<core::Tensor>(t2);
    return ITensorSPtr(core::add(tensor1, tensor2));
}

ITensorSPtr operator+(ITensorSPtr t1, ITensorSPtr t2) { return add(t1, t2); }

}  // namespace dll
