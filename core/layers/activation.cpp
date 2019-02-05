#include "activation.h"

#include "abstractTensor.h"
#include "graph.h"
#include "graphdl_ops.h"

#include <cassert>
#include <cmath>
#include <utility>

namespace graphdl
{
namespace core
{
namespace layers
{
namespace
{
template <Activation act>
float activation(float x);

template <>
float activation<Activation::kRELU>(float x)
{
    return x >= 0. ? x : 0.;
}
template <>
float activation<Activation::kSIGMOID>(float x)
{
    return 1. / (1. + std::exp(-x));
}
template <>
float activation<Activation::kTANH>(float x)
{
    return std::tanh(x);
}
template <>
float activation<Activation::kSQUARE>(float x)
{
    return x * x;
}
template <>
float activation<Activation::kABS>(float x)
{
    return std::abs(x);
}
template <>
float activation<Activation::kNEG>(float x)
{
    return -x;
}
template <>
float activation<Activation::kRECIPROCAL>(float x)
{
    return 1. / x;
}
template <>
float activation<Activation::kLOG>(float x)
{
    return std::log(x);
}
template <>
float activation<Activation::kSQRT>(float x)
{
    return std::sqrt(x);
}
template <>
float activation<Activation::kEXP>(float x)
{
    return std::exp(x);
}

template <Activation act>
void activationHost(const float* x, float* y, size_t size)
{
    for (std::size_t i = 0; i < size; ++i) y[i] = activation<act>(x[i]);
}

template <Activation act>
float activationGradient(float x, float o);
template <>
float activationGradient<Activation::kRELU>(float x, float /* o */)
{
    return x >= 0. ? 1. : 0.;
}
template <>
float activationGradient<Activation::kSIGMOID>(float /* x */, float o)
{
    return o * (1. - o);
}
template <>
float activationGradient<Activation::kTANH>(float /* x */, float o)
{
    return 1. - o * o;
}
template <>
float activationGradient<Activation::kSQUARE>(float x, float /* o */)
{
    return 2. * x;
}
template <>
float activationGradient<Activation::kABS>(float x, float /* o */)
{
    return x >= 0. ? 1. : -1;
}
template <>
float activationGradient<Activation::kNEG>(float /* x */, float /* o */)
{
    return -1;
}
template <>
float activationGradient<Activation::kRECIPROCAL>(float /* x */, float o)
{
    return -1. * o * o;
}
template <>
float activationGradient<Activation::kLOG>(float x, float /* o */)
{
    return 1. / x;
}
template <>
float activationGradient<Activation::kSQRT>(float /* x */, float o)
{
    return 1. / (2 * o);
}
template <>
float activationGradient<Activation::kEXP>(float /* x */, float o)
{
    return o;
}

template <Activation act>
void activationGradientHost(const float* x, const float* y, const float* yGrad,
                            float* xGrad, size_t size)
{
    for (size_t i = 0; i < size; ++i)
        xGrad[i] = yGrad[i] * activationGradient<act>(x[i], y[i]);
}

}  // namespace

void runActivationHost(const float* x, float* y, size_t size, Activation op)
{
    switch (op)
    {
    case Activation::kRELU:
        activationHost<Activation::kRELU>(x, y, size);
        return;
    case Activation::kSIGMOID:
        activationHost<Activation::kSIGMOID>(x, y, size);
        return;
    case Activation::kTANH:
        activationHost<Activation::kTANH>(x, y, size);
        return;
    case Activation::kSQUARE:
        activationHost<Activation::kSQUARE>(x, y, size);
        return;
    case Activation::kABS: activationHost<Activation::kABS>(x, y, size); return;
    case Activation::kNEG: activationHost<Activation::kNEG>(x, y, size); return;
    case Activation::kRECIPROCAL:
        activationHost<Activation::kRECIPROCAL>(x, y, size);
        return;
    case Activation::kLOG: activationHost<Activation::kLOG>(x, y, size); return;
    case Activation::kSQRT:
        activationHost<Activation::kSQRT>(x, y, size);
        return;
    case Activation::kEXP: activationHost<Activation::kEXP>(x, y, size); return;
    }
}

void runActivationGradientHost(const float* x, const float* y,
                               const float* yGrad, float* xGrad, size_t size,
                               Activation op)
{
    switch (op)
    {
    case Activation::kRELU:
        activationGradientHost<Activation::kRELU>(x, y, yGrad, xGrad, size);
        return;
    case Activation::kSIGMOID:
        activationGradientHost<Activation::kSIGMOID>(x, y, yGrad, xGrad, size);
        return;
    case Activation::kTANH:
        activationGradientHost<Activation::kTANH>(x, y, yGrad, xGrad, size);
        return;
    case Activation::kSQUARE:
        activationGradientHost<Activation::kSQUARE>(x, y, yGrad, xGrad, size);
        return;
    case Activation::kABS:
        activationGradientHost<Activation::kABS>(x, y, yGrad, xGrad, size);
        return;
    case Activation::kNEG:
        activationGradientHost<Activation::kNEG>(x, y, yGrad, xGrad, size);
        return;
    case Activation::kRECIPROCAL:
        activationGradientHost<Activation::kRECIPROCAL>(x, y, yGrad, xGrad,
                                                        size);
        return;
    case Activation::kLOG:
        activationGradientHost<Activation::kLOG>(x, y, yGrad, xGrad, size);
        return;
    case Activation::kSQRT:
        activationGradientHost<Activation::kSQRT>(x, y, yGrad, xGrad, size);
        return;
    case Activation::kEXP:
        activationGradientHost<Activation::kEXP>(x, y, yGrad, xGrad, size);
        return;
    }
}

ActivationLayer::ActivationLayer(ID id, const Tensor::SPtr& t, Activation op)
    : DifferentiableLayer(id, {t},
                          {createTensor("", t->getShape(), t->getType())}),
      mOp(op)
{
}

void ActivationLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr xTensor = mInputs[0].lock();
    xTensor->eval(inputs);

    float* x = xTensor->getMemory().getValues();
    float* y = mOutputs[0]->getMemory().getValues();
    size_t size = xTensor->getMemory().getCount();

    if (xTensor->getType() == MemoryType::kHOST_MEMORY)
        runActivationHost(x, y, size, mOp);
#ifdef CUDA_AVAILABLE
    else  // xTensor->getType() == MemoryType::kDEVICE_MEMORY
        cuda::runActivationDevice(x, y, size, mOp);
#endif
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

ActivationGradientLayer::ActivationGradientLayer(ID id, const Tensor::SPtr& in,
                                                 const Tensor::SPtr& out,
                                                 const Tensor::SPtr& outGrad,
                                                 Activation op)
    : Layer(id, {in, out, outGrad},
            {createTensor("", in->getShape(), in->getType())}),
      mOp(op)
{
    assert(in->getShape() == out->getShape());
    assert(out->getShape() == outGrad->getShape());

    assert(in->getType() == out->getType());
    assert(in->getType() == outGrad->getType());
}

void ActivationGradientLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr xTensor = mInputs[0].lock();
    Tensor::SPtr yTensor = mInputs[1].lock();
    Tensor::SPtr yGradTensor = mInputs[2].lock();
    xTensor->eval(inputs);
    yTensor->eval(inputs);
    yGradTensor->eval(inputs);

    float* x = xTensor->getMemory().getValues();
    float* y = yTensor->getMemory().getValues();
    float* yGrad = yGradTensor->getMemory().getValues();
    float* xGrad = mOutputs[0]->getMemory().getValues();
    size_t size = xTensor->getMemory().getCount();

    if (xTensor->getType() == MemoryType::kHOST_MEMORY)
        runActivationGradientHost(x, y, yGrad, xGrad, size, mOp);
#ifdef CUDA_AVAILABLE
    else  // xTensor->getType() == MemoryType::kDEVICE_MEMORY
        cuda::runActivationGradientDevice(x, y, yGrad, xGrad, size, mOp);
#endif
}

}  // namespace layers

Tensor::SPtr createActivation(Tensor::SPtr t, layers::Activation op)
{
    Layer::SPtr layer = createLayer<layers::ActivationLayer>(std::move(t), op);
    return layer->getOutputs()[0];
}

Tensor::SPtr relu(Tensor::SPtr t)
{
    return createActivation(std::move(t), layers::Activation::kRELU);
}
Tensor::SPtr sigmoid(Tensor::SPtr t)
{
    return createActivation(std::move(t), layers::Activation::kSIGMOID);
}
Tensor::SPtr tanh(Tensor::SPtr t)
{
    return createActivation(std::move(t), layers::Activation::kTANH);
}
Tensor::SPtr square(Tensor::SPtr t)
{
    return createActivation(std::move(t), layers::Activation::kSQUARE);
}
Tensor::SPtr abs(Tensor::SPtr t)
{
    return createActivation(std::move(t), layers::Activation::kABS);
}
Tensor::SPtr neg(Tensor::SPtr t)
{
    return createActivation(std::move(t), layers::Activation::kNEG);
}
Tensor::SPtr reciprocal(Tensor::SPtr t)
{
    return createActivation(std::move(t), layers::Activation::kRECIPROCAL);
}
Tensor::SPtr log(Tensor::SPtr t)
{
    return createActivation(std::move(t), layers::Activation::kLOG);
}
Tensor::SPtr sqrt(Tensor::SPtr t)
{
    return createActivation(std::move(t), layers::Activation::kSQRT);
}
Tensor::SPtr exp(Tensor::SPtr t)
{
    return createActivation(std::move(t), layers::Activation::kEXP);
}

}  // namespace core

ITensorPtr relu(const ITensorPtr& t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::relu(tensor->get()));
}

ITensorPtr sigmoid(const ITensorPtr& t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::sigmoid(tensor->get()));
}

ITensorPtr tanh(const ITensorPtr& t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::tanh(tensor->get()));
}

ITensorPtr square(const ITensorPtr& t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::square(tensor->get()));
}

ITensorPtr abs(const ITensorPtr& t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::abs(tensor->get()));
}

ITensorPtr neg(const ITensorPtr& t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::neg(tensor->get()));
}

ITensorPtr reciprocal(const ITensorPtr& t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::reciprocal(tensor->get()));
}

ITensorPtr log(const ITensorPtr& t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::log(tensor->get()));
}

ITensorPtr sqrt(const ITensorPtr& t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::sqrt(tensor->get()));
}

ITensorPtr exp(const ITensorPtr& t)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::exp(tensor->get()));
}

}  // namespace graphdl
