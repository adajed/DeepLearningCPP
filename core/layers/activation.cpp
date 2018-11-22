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
float op(float x);
template <>
float op<Activation::kRELU>(float x)
{
    return x >= 0. ? x : 0.;
}
template <>
float op<Activation::kSIGMOID>(float x)
{
    return 1. / (1. + std::exp(-x));
}
template <>
float op<Activation::kTANH>(float x)
{
    return std::tanh(x);
}
template <>
float op<Activation::kSQUARE>(float x)
{
    return x * x;
}
template <>
float op<Activation::kABS>(float x)
{
    return std::abs(x);
}
template <>
float op<Activation::kNEG>(float x)
{
    return -x;
}
template <>
float op<Activation::kRECIPROCAL>(float x)
{
    return 1. / x;
}
template <>
float op<Activation::kLOG>(float x)
{
    return std::log(x);
}
template <>
float op<Activation::kSQRT>(float x)
{
    return std::sqrt(x);
}

template <Activation act>
void activation(std::size_t size, float* x, float* y)
{
    for (std::size_t i = 0; i < size; ++i) y[i] = op<act>(x[i]);
}

void runActivationHost(std::size_t size, float* x, float* y, Activation op)
{
    switch (op)
    {
    case Activation::kRELU: activation<Activation::kRELU>(size, x, y); return;
    case Activation::kSIGMOID:
        activation<Activation::kSIGMOID>(size, x, y);
        return;
    case Activation::kTANH: activation<Activation::kTANH>(size, x, y); return;
    case Activation::kSQUARE:
        activation<Activation::kSQUARE>(size, x, y);
        return;
    case Activation::kABS: activation<Activation::kABS>(size, x, y); return;
    case Activation::kNEG: activation<Activation::kNEG>(size, x, y); return;
    case Activation::kRECIPROCAL:
        activation<Activation::kRECIPROCAL>(size, x, y);
        return;
    case Activation::kLOG: activation<Activation::kLOG>(size, x, y); return;
    case Activation::kSQRT: activation<Activation::kSQRT>(size, x, y); return;
    }
}

template <Activation act>
float opGrad(float x, float o);
template <>
float opGrad<Activation::kRELU>(float x, float /* o */)
{
    return x >= 0. ? 1. : 0.;
}
template <>
float opGrad<Activation::kSIGMOID>(float /* x */, float o)
{
    return o * (1. - o);
}
template <>
float opGrad<Activation::kTANH>(float /* x */, float o)
{
    return 1. - o * o;
}
template <>
float opGrad<Activation::kSQUARE>(float x, float /* o */)
{
    return 2. * x;
}
template <>
float opGrad<Activation::kABS>(float x, float /* o */)
{
    return x >= 0. ? 1. : -1;
}
template <>
float opGrad<Activation::kNEG>(float /* x */, float /* o */)
{
    return -1;
}
template <>
float opGrad<Activation::kRECIPROCAL>(float /* x */, float o)
{
    return -1. * o * o;
}
template <>
float opGrad<Activation::kLOG>(float x, float /* o */)
{
    return 1. / x;
}
template <>
float opGrad<Activation::kSQRT>(float /* x */, float o)
{
    return -1. / (2 * o);
}

template <Activation act>
void activationGradient(std::size_t size, float* x, float* y, float* yGrad,
                        float* xGrad)
{
    for (std::size_t i = 0; i < size; ++i)
        xGrad[i] = yGrad[i] * opGrad<act>(x[i], y[i]);
}

void runActivationGradientHost(std::size_t size, float* x, float* y,
                               float* yGrad, float* xGrad, Activation op)
{
    switch (op)
    {
    case Activation::kRELU:
        activationGradient<Activation::kRELU>(size, x, y, yGrad, xGrad);
        return;
    case Activation::kSIGMOID:
        activationGradient<Activation::kSIGMOID>(size, x, y, yGrad, xGrad);
        return;
    case Activation::kTANH:
        activationGradient<Activation::kTANH>(size, x, y, yGrad, xGrad);
        return;
    case Activation::kSQUARE:
        activationGradient<Activation::kSQUARE>(size, x, y, yGrad, xGrad);
        return;
    case Activation::kABS:
        activationGradient<Activation::kABS>(size, x, y, yGrad, xGrad);
        return;
    case Activation::kNEG:
        activationGradient<Activation::kNEG>(size, x, y, yGrad, xGrad);
        return;
    case Activation::kRECIPROCAL:
        activationGradient<Activation::kRECIPROCAL>(size, x, y, yGrad, xGrad);
        return;
    case Activation::kLOG:
        activationGradient<Activation::kLOG>(size, x, y, yGrad, xGrad);
        return;
    case Activation::kSQRT:
        activationGradient<Activation::kSQRT>(size, x, y, yGrad, xGrad);
        return;
    }
}

}  // namespace

ActivationLayer::ActivationLayer(ID id, const Tensor::SPtr& t, Activation op)
    : DifferentiableLayer(id, {t},
                          {createTensor("", t->getShape(), t->getType())}),
      mOp(op)
{
}

void ActivationLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr in = mInputs[0].lock();
    in->eval(inputs);

    float* input = in->getMemory().getValues();
    float* output = mOutputs[0]->getMemory().getValues();
    std::size_t size = in->getMemory().getCount();

    if (in->getType() == MemoryType::kHOST_MEMORY)
        runActivationHost(size, input, output, mOp);
#ifdef CUDA_AVAILABLE
    else  // in->getType() == MemoryType::kDEVICE_MEMORY
        cuda::runActivationDevice(size, input, output, mOp);
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

    if (in->getType() == MemoryType::kHOST_MEMORY)
        runActivationGradientHost(size, input, output, outputGrad, gradient,
                                  mOp);
#ifdef CUDA_AVAILABLE
    else  // in->getType() == MemoryType::kDEVICE_MEMORY
        cuda::runActivationGradientDevice(size, input, output, outputGrad,
                                          gradient, mOp);
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

}  // namespace graphdl
