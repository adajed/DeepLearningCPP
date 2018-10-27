#include "activation.h"
#include <assert.h>
#include <cmath>
#include "dll_ops.h"
#include "graph.h"

namespace dll
{
namespace core
{
namespace layers
{
std::vector<Tensor::SPtr> createOutputs(Tensor::SPtr t)
{
    return {createTensor("", t->shape())};
}

ActivationOper::ActivationOper(Tensor::SPtr t, Activation op)
    : GradientOper({t}, createOutputs(t)), mOp(op)
{
    switch (op)
    {
        case Activation::kRELU:
            mFun = [](float x) { return x > 0. ? x : 0.; };
            break;
        case Activation::kSIGMOID:
            mFun = [](float x) { return 1. / (1. + std::exp(-x)); };
            break;
        case Activation::kTANH:
            mFun = [](float x) { return std::tanh(x); };
            break;
        case Activation::kSQUARE:
            mFun = [](float x) { return x * x; };
            break;
        case Activation::kSQRT:
            mFun = [](float x) { return std::sqrt(x); };
            break;
        case Activation::kABS:
            mFun = [](float x) { return std::abs(x); };
            break;
        case Activation::kNEG:
            mFun = [](float x) { return -x; };
            break;
        case Activation::kRECIPROCAL:
            mFun = [](float x) { return 1. / x; };
            break;
    }
}

void ActivationOper::executeOper(const InputDict& inputs)
{
    Tensor::SPtr in = mInputs[0].lock();
    in->exec(inputs);

    Memory input = in->getMemory();
    Memory output = mOutputs[0]->getMemory();

    for (std::size_t pos = 0; pos < input.count(); ++pos)
        output[pos] = mFun(output[pos]);
}

GradientOper::TensorMap ActivationOper::gradients(Tensor::SPtr out,
                                                  Tensor::SPtr outGrad)
{
    assert(out == mOutputs[0]);

    std::vector<Tensor::SPtr> inputs = getInputs();
    Oper::SPtr oper =
        std::make_shared<ActivationGradientOper>(inputs[0], out, outGrad, mOp);
    core::getDefaultGraph()->insertOperation(oper);
    std::vector<Tensor::SPtr> grads = oper->getOutputs();

    return {{inputs[0], grads[0]}};
}

std::vector<Tensor::SPtr> createGradientOutputs(Tensor::SPtr t)
{
    return {createTensor("", t->shape())};
}

ActivationGradientOper::ActivationGradientOper(Tensor::SPtr in,
                                               Tensor::SPtr out,
                                               Tensor::SPtr outGrad,
                                               Activation op)
    : Oper({in, out, outGrad}, createGradientOutputs(in))
{
    assert(in->shape() == out->shape());
    assert(out->shape() == outGrad->shape());

    switch (op)
    {
        case Activation::kRELU:
            mFun = [](float x, float o) { return x > 0. ? 1. : 0.; };
            break;
        case Activation::kSIGMOID:
            mFun = [](float x, float o) { return o * (1. - o); };
            break;
        case Activation::kTANH:
            mFun = [](float x, float o) { return 1. - o * o; };
            break;
        case Activation::kSQUARE:
            mFun = [](float x, float o) { return 2 * x; };
            break;
        case Activation::kSQRT:
            mFun = [](float x, float o) { return 1. / (2 * o); };
            break;
        case Activation::kABS:
            mFun = [](float x, float o) { return x > 0 ? 1. : -1.; };
            break;
        case Activation::kNEG:
            mFun = [](float x, float o) { return -1; };
            break;
        case Activation::kRECIPROCAL:
            mFun = [](float x, float o) { return -1. * o * o; };
            break;
    }
}

void ActivationGradientOper::executeOper(const InputDict& inputs)
{
    Tensor::SPtr in = mInputs[0].lock();
    Tensor::SPtr out = mInputs[1].lock();
    Tensor::SPtr outGrad = mInputs[2].lock();
    in->exec(inputs);
    out->exec(inputs);
    outGrad->exec(inputs);

    Memory input = in->getMemory();
    Memory output = out->getMemory();
    Memory outputGrad = outGrad->getMemory();
    Memory gradient = mOutputs[0]->getMemory();

    for (std::size_t pos = 0; pos < input.count(); ++pos)
        gradient[pos] = outputGrad[pos] * mFun(input[pos], output[pos]);
}

Tensor::SPtr createActivation(Tensor::SPtr t, Activation op)
{
    Oper::SPtr oper = std::make_shared<ActivationOper>(t, op);
    core::getDefaultGraph()->insertOperation(oper);
    return oper->getOutputs()[0];
}

}  // namespace layers

Tensor::SPtr relu(Tensor::SPtr t)
{
    return createActivation(t, layers::Activation::kRELU);
}
Tensor::SPtr sigmoid(Tensor::SPtr t)
{
    return createActivation(t, layers::Activation::kSIGMOID);
}
Tensor::SPtr tanh(Tensor::SPtr t)
{
    return createActivation(t, layers::Activation::kTANH);
}
Tensor::SPtr square(Tensor::SPtr t)
{
    return createActivation(t, layers::Activation::kSQUARE);
}
Tensor::SPtr sqrt(Tensor::SPtr t)
{
    return createActivation(t, layers::Activation::kSQRT);
}
Tensor::SPtr abs(Tensor::SPtr t)
{
    return createActivation(t, layers::Activation::kABS);
}
Tensor::SPtr neg(Tensor::SPtr t)
{
    return createActivation(t, layers::Activation::kNEG);
}
Tensor::SPtr reciprocal(Tensor::SPtr t)
{
    return createActivation(t, layers::Activation::kRECIPROCAL);
}

}  // namespace core

ITensorSPtr relu(ITensorSPtr t)
{
    core::Tensor::SPtr tensor = std::static_pointer_cast<core::Tensor>(t);
    return ITensorSPtr(core::relu(tensor));
}

ITensorSPtr sigmoid(ITensorSPtr t)
{
    core::Tensor::SPtr tensor = std::static_pointer_cast<core::Tensor>(t);
    return ITensorSPtr(core::sigmoid(tensor));
}

ITensorSPtr tanh(ITensorSPtr t)
{
    core::Tensor::SPtr tensor = std::static_pointer_cast<core::Tensor>(t);
    return ITensorSPtr(core::tanh(tensor));
}

ITensorSPtr square(ITensorSPtr t)
{
    core::Tensor::SPtr tensor = std::static_pointer_cast<core::Tensor>(t);
    return ITensorSPtr(core::square(tensor));
}

ITensorSPtr sqrt(ITensorSPtr t)
{
    core::Tensor::SPtr tensor = std::static_pointer_cast<core::Tensor>(t);
    return ITensorSPtr(core::sqrt(tensor));
}

ITensorSPtr abs(ITensorSPtr t)
{
    core::Tensor::SPtr tensor = std::static_pointer_cast<core::Tensor>(t);
    return ITensorSPtr(core::abs(tensor));
}

ITensorSPtr neg(ITensorSPtr t)
{
    core::Tensor::SPtr tensor = std::static_pointer_cast<core::Tensor>(t);
    return ITensorSPtr(core::neg(tensor));
}

ITensorSPtr reciprocal(ITensorSPtr t)
{
    core::Tensor::SPtr tensor = std::static_pointer_cast<core::Tensor>(t);
    return ITensorSPtr(core::reciprocal(tensor));
}

}  // namespace dll
