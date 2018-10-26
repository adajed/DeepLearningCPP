#include "mulOper.h"
#include "dll_errors.h"
#include "dll_ops.h"
#include "graph.h"

namespace dll
{
namespace core
{
namespace layers
{
MulGradientOper::MulGradientOper(Tensor::SPtr in1, Tensor::SPtr in2,
                                 Tensor::SPtr out, Tensor::SPtr outGrad)
    : Oper({in1, in2, out, outGrad}, createOutputs(in1, in2))
{
}

std::vector<Tensor::SPtr> MulGradientOper::createOutputs(Tensor::SPtr in1,
                                                         Tensor::SPtr in2)
{
    Tensor::SPtr grad1 = std::make_shared<Tensor>("", in1->shape());
    Tensor::SPtr grad2 = std::make_shared<Tensor>("", in2->shape());
    return {grad1, grad2};
}

void MulGradientOper::executeOper(const InputDict& inputs)
{
    Tensor::SPtr input1 = mInputs[0].lock();
    Tensor::SPtr input2 = mInputs[1].lock();
    Tensor::SPtr outputGrad = mInputs[3].lock();
    input1->exec(inputs);
    input2->exec(inputs);
    outputGrad->exec(inputs);

    Memory in1 = input1->getMemory();
    Memory in2 = input2->getMemory();
    Memory outGrad = outputGrad->getMemory();

    Memory gradient1 = mOutputs[0]->getMemory();
    Memory gradient2 = mOutputs[1]->getMemory();

    for (std::size_t i = 0; i < in1.count(); ++i)
    {
        gradient1[i] = in2[i] * outGrad[i];
        gradient2[i] = in1[i] * outGrad[i];
    }
}

GradientOper::TensorMap MulOper::gradients(Tensor::SPtr output,
                                           Tensor::SPtr outputGrad)
{
    assert(output == mOutputs[0]);
    std::vector<Tensor::SPtr> inputs = getInputs();

    Oper::SPtr gradOper = std::make_shared<MulGradientOper>(
        inputs[0], inputs[1], output, outputGrad);
    getDefaultGraph()->insertOperation(gradOper);
    std::vector<Tensor::SPtr> grads = gradOper->getOutputs();

    return {{inputs[0], grads[0]}, {inputs[1], grads[1]}};
}

}  // namespace layers

Tensor::SPtr mul(Tensor::SPtr t1, Tensor::SPtr t2)
{
    if (t1->shape() != t2->shape()) throw errors::NotMatchingShapesError();
    Oper::SPtr oper = std::make_shared<layers::MulOper>(t1, t2);
    getDefaultGraph()->insertOperation(oper);
    return oper->getOutputs()[0];
}

Tensor::SPtr operator*(Tensor::SPtr t1, Tensor::SPtr t2) { return mul(t1, t2); }

}  // namespace core

ITensorSPtr mul(ITensorSPtr t1, ITensorSPtr t2)
{
    core::Tensor::SPtr tensor1 = std::static_pointer_cast<core::Tensor>(t1);
    core::Tensor::SPtr tensor2 = std::static_pointer_cast<core::Tensor>(t2);
    return ITensorSPtr(core::mul(tensor1, tensor2));
}

ITensorSPtr operator*(ITensorSPtr t1, ITensorSPtr t2) { return mul(t1, t2); }

}  // namespace dll
