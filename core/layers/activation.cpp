#include "activation.h"
#include <assert.h>
#include <cmath>
#include "abstractTensor.h"
#include "graph.h"
#include "graphdl_ops.h"

namespace graphdl
{
namespace core
{
namespace layers
{
ActivationLayer::ActivationLayer(ID id, Tensor::SPtr t, Activation op)
    : DifferentiableLayer(id, {t}, {createTensor("", t->getShape(), t->getType())}), mOp(op)
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
        case Activation::kABS:
            mFun = [](float x) { return std::abs(x); };
            break;
        case Activation::kNEG:
            mFun = [](float x) { return -x; };
            break;
        case Activation::kRECIPROCAL:
            mFun = [](float x) { return 1. / x; };
            break;
        case Activation::kLOG:
            mFun = [](float x) { return std::log(x); };
            break;
    }
}

void ActivationLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr in = mInputs[0].lock();
    in->eval(inputs);

    float* input = in->getMemory().getValues();
    float* output = mOutputs[0]->getMemory().getValues();
    std::size_t size = in->getMemory().getCount();

    for (std::size_t pos = 0; pos < size; ++pos) output[pos] = mFun(input[pos]);
}

Layer::TensorMap ActivationLayer::gradients(Tensor::SPtr out,
                                            Tensor::SPtr outGrad)
{
    assert(out == mOutputs[0]);

    std::vector<Tensor::SPtr> inputs = getInputs();
    Layer::SPtr layer =
        createLayer<ActivationGradientLayer>(inputs[0], out, outGrad, mOp);

    std::vector<Tensor::SPtr> grads = layer->getOutputs();
    return {{inputs[0], grads[0]}};
}

ActivationGradientLayer::ActivationGradientLayer(ID id, Tensor::SPtr in,
                                                 Tensor::SPtr out,
                                                 Tensor::SPtr outGrad,
                                                 Activation op)
    : Layer(id, {in, out, outGrad}, {createTensor("", in->getShape(), outGrad->getType())})
{
    assert(in->getShape() == out->getShape());
    assert(out->getShape() == outGrad->getShape());

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
        case Activation::kABS:
            mFun = [](float x, float o) { return x > 0 ? 1. : -1.; };
            break;
        case Activation::kNEG:
            mFun = [](float x, float o) { return -1; };
            break;
        case Activation::kRECIPROCAL:
            mFun = [](float x, float o) { return -1. * o * o; };
            break;
        case Activation::kLOG:
            mFun = [](float x, float o) { return 1. / x; };
            break;
    }
}

void ActivationGradientLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr in = mInputs[0].lock();
    Tensor::SPtr out = mInputs[1].lock();
    Tensor::SPtr outGrad = mInputs[2].lock();
    in->eval(inputs);
    out->eval(inputs);
    outGrad->eval(inputs);

    float* input = in->getMemory().getValues();
    float* output = out->getMemory().getValues();
    float* outputGrad = outGrad->getMemory().getValues();
    float* gradient = mOutputs[0]->getMemory().getValues();
    std::size_t size = in->getMemory().getCount();

    for (std::size_t pos = 0; pos < size; ++pos)
        gradient[pos] = outputGrad[pos] * mFun(input[pos], output[pos]);
}

}  // namespace layers

Tensor::SPtr createActivation(Tensor::SPtr t, layers::Activation op)
{
    Layer::SPtr layer = createLayer<layers::ActivationLayer>(t, op);
    return layer->getOutputs()[0];
}

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
Tensor::SPtr log(Tensor::SPtr t)
{
    return createActivation(t, layers::Activation::kLOG);
}

}  // namespace core

ITensorPtr relu(ITensorPtr t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::relu(tensor->get()));
}

ITensorPtr sigmoid(ITensorPtr t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::sigmoid(tensor->get()));
}

ITensorPtr tanh(ITensorPtr t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::tanh(tensor->get()));
}

ITensorPtr square(ITensorPtr t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::square(tensor->get()));
}

ITensorPtr abs(ITensorPtr t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::abs(tensor->get()));
}

ITensorPtr neg(ITensorPtr t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::neg(tensor->get()));
}

ITensorPtr reciprocal(ITensorPtr t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::reciprocal(tensor->get()));
}

ITensorPtr log(ITensorPtr t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::log(tensor->get()));
}

}  // namespace graphdl
