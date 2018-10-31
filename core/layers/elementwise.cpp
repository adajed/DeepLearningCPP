#include <assert.h>
#include "dll_errors.h"
#include "dll_ops.h"
#include "elementwiseOper.h"
#include "graph.h"

namespace dll
{
namespace core
{
namespace layers
{
ElementwiseOper::ElementwiseOper(Tensor::SPtr t1, Tensor::SPtr t2,
                                 Elementwise op)
    : GradientOper({t1, t2}, createOutputs(t1, t2)), mOp(op)
{
    switch (op)
    {
        case Elementwise::kADD:
            mFun = [](float f1, float f2) { return f1 + f2; };
            break;
        case Elementwise::kSUB:
            mFun = [](float f1, float f2) { return f1 - f2; };
            break;
        case Elementwise::kMUL:
            mFun = [](float f1, float f2) { return f1 * f2; };
            break;
        case Elementwise::kDIV:
            mFun = [](float f1, float f2) { return f1 / f2; };
            break;
    }
}

std::vector<Tensor::SPtr> ElementwiseOper::createOutputs(Tensor::SPtr t1,
                                                         Tensor::SPtr t2)
{
    assert(t1->shape() == t2->shape());
    return {std::make_shared<Tensor>("", t1->shape())};
}

void ElementwiseOper::executeOper(const InputDict& inputs)
{
    Tensor::SPtr i0 = mInputs[0].lock();
    Tensor::SPtr i1 = mInputs[1].lock();

    i0->exec(inputs);
    i1->exec(inputs);

    Memory input0 = i0->getMemory();
    Memory input1 = i1->getMemory();
    Memory output = mOutputs[0]->getMemory();

    for (std::size_t i = 0; i < output.count(); ++i)
        output[i] = mFun(input0[i], input1[i]);
}

GradientOper::TensorMap ElementwiseOper::gradients(Tensor::SPtr output,
                                                   Tensor::SPtr outputGrad)
{
    assert(output == mOutputs[0]);

    std::vector<Tensor::SPtr> inputs = getInputs();

    Oper::SPtr gradOper = Oper::SPtr(std::make_shared<ElementwiseGradientOper>(
        inputs[0], inputs[1], output, outputGrad, mOp));
    getDefaultGraph()->insertOperation(gradOper);
    std::vector<Tensor::SPtr> grads = gradOper->getOutputs();

    return {{inputs[0], grads[0]}, {inputs[1], grads[1]}};
}

ElementwiseGradientOper::ElementwiseGradientOper(Tensor::SPtr t1,
                                                 Tensor::SPtr t2,
                                                 Tensor::SPtr out,
                                                 Tensor::SPtr outGrad,
                                                 Elementwise op)
    : Oper({t1, t2, out, outGrad}, createOutputs(t1, t2)), mOp(op)
{
    switch (op)
    {
        case Elementwise::kADD:
            mFun1 = [](float f1, float f2) { return 1.; };
            mFun2 = [](float f1, float f2) { return 1.; };
            break;
        case Elementwise::kSUB:
            mFun1 = [](float f1, float f2) { return 1.; };
            mFun2 = [](float f1, float f2) { return -1.; };
            break;
        case Elementwise::kMUL:
            mFun1 = [](float f1, float f2) { return f2; };
            mFun2 = [](float f1, float f2) { return f1; };
            break;
        case Elementwise::kDIV:
            mFun1 = [](float f1, float f2) { return 1. / f2; };
            mFun2 = [](float f1, float f2) { return -f1 / (f2 * f2); };
            break;
    }
};

std::vector<Tensor::SPtr> ElementwiseGradientOper::createOutputs(
    Tensor::SPtr t1, Tensor::SPtr t2)
{
    Tensor::SPtr grad1 = std::make_shared<Tensor>("", t1->shape());
    Tensor::SPtr grad2 = std::make_shared<Tensor>("", t2->shape());
    return {grad1, grad2};
}

void ElementwiseGradientOper::executeOper(const InputDict& inputs)
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
        gradient1[i] = outGrad[i] * mFun1(in1[i], in2[i]);
        gradient2[i] = outGrad[i] * mFun2(in1[i], in2[i]);
    }
}

Tensor::SPtr createElementwise(Tensor::SPtr t1, Tensor::SPtr t2, Elementwise op)
{
    if (t1->shape() != t2->shape()) throw errors::NotMatchingShapesError();
    Oper::SPtr oper = std::make_shared<ElementwiseOper>(t1, t2, op);
    getDefaultGraph()->insertOperation(oper);
    return oper->getOutputs()[0];
}

}  // namespace layers

Tensor::SPtr add(Tensor::SPtr t1, Tensor::SPtr t2)
{
    return layers::createElementwise(t1, t2, layers::Elementwise::kADD);
}

Tensor::SPtr sub(Tensor::SPtr t1, Tensor::SPtr t2)
{
    return layers::createElementwise(t1, t2, layers::Elementwise::kSUB);
}

Tensor::SPtr mul(Tensor::SPtr t1, Tensor::SPtr t2)
{
    return layers::createElementwise(t1, t2, layers::Elementwise::kMUL);
}

Tensor::SPtr div(Tensor::SPtr t1, Tensor::SPtr t2)
{
    return layers::createElementwise(t1, t2, layers::Elementwise::kDIV);
}

Tensor::SPtr operator+(Tensor::SPtr t1, Tensor::SPtr t2) { return add(t1, t2); }

Tensor::SPtr operator-(Tensor::SPtr t1, Tensor::SPtr t2) { return sub(t1, t2); }

Tensor::SPtr operator*(Tensor::SPtr t1, Tensor::SPtr t2) { return mul(t1, t2); }

Tensor::SPtr operator/(Tensor::SPtr t1, Tensor::SPtr t2) { return div(t1, t2); }

}  // namespace core

ITensorSPtr add(ITensorSPtr t1, ITensorSPtr t2)
{
    core::Tensor::SPtr tensor1 = std::static_pointer_cast<core::Tensor>(t1);
    core::Tensor::SPtr tensor2 = std::static_pointer_cast<core::Tensor>(t2);
    return ITensorSPtr(core::add(tensor1, tensor2));
}

ITensorSPtr sub(ITensorSPtr t1, ITensorSPtr t2)
{
    core::Tensor::SPtr tensor1 = std::static_pointer_cast<core::Tensor>(t1);
    core::Tensor::SPtr tensor2 = std::static_pointer_cast<core::Tensor>(t2);
    return ITensorSPtr(core::sub(tensor1, tensor2));
}

ITensorSPtr mul(ITensorSPtr t1, ITensorSPtr t2)
{
    core::Tensor::SPtr tensor1 = std::static_pointer_cast<core::Tensor>(t1);
    core::Tensor::SPtr tensor2 = std::static_pointer_cast<core::Tensor>(t2);
    return ITensorSPtr(core::mul(tensor1, tensor2));
}

ITensorSPtr div(ITensorSPtr t1, ITensorSPtr t2)
{
    core::Tensor::SPtr tensor1 = std::static_pointer_cast<core::Tensor>(t1);
    core::Tensor::SPtr tensor2 = std::static_pointer_cast<core::Tensor>(t2);
    return ITensorSPtr(core::div(tensor1, tensor2));
}

ITensorSPtr operator+(ITensorSPtr t1, ITensorSPtr t2) { return add(t1, t2); }

ITensorSPtr operator-(ITensorSPtr t1, ITensorSPtr t2) { return sub(t1, t2); }

ITensorSPtr operator*(ITensorSPtr t1, ITensorSPtr t2) { return mul(t1, t2); }

ITensorSPtr operator/(ITensorSPtr t1, ITensorSPtr t2) { return div(t1, t2); }

}  // namespace dll
