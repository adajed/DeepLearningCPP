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
template <>
float activation<Activation::kLEAKY_RELU>(float x)
{
    return x >= 0. ? x : 0.01 * x;
}
template <>
float activation<Activation::kRELU_6>(float x)
{
    return x >= 0. ? (x <= 6. ? x : 6.) : 0.;
}
template <>
float activation<Activation::kELU>(float x)
{
    return x >= 0. ? x : std::exp(x) - 1.;
}
template <>
float activation<Activation::kSOFTPLUS>(float x)
{
    return std::log(std::exp(x) + 1.);
}
template <>
float activation<Activation::kSOFTSIGN>(float x)
{
    return x / (std::abs(x) + 1.);
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
template <>
float activationGradient<Activation::kLEAKY_RELU>(float x, float /* o */)
{
    return x >= 0. ? 1. : 0.01;
}
template <>
float activationGradient<Activation::kRELU_6>(float x, float /* o */)
{
    return x >= 0. ? (x <= 6. ? 1. : 0.) : 0.;
}
template <>
float activationGradient<Activation::kELU>(float x, float o)
{
    return x >= 0. ? 1. : o + 1.;
}
template <>
float activationGradient<Activation::kSOFTPLUS>(float x, float /* o */)
{
    float v = std::exp(x);
    return v / (v + 1.);
}
template <>
float activationGradient<Activation::kSOFTSIGN>(float x, float /* o */)
{
    float v = std::abs(x) + 1.;
    return 1. / (v * v);
}

template <Activation act>
void activationGradientHost(const float* x, const float* y, const float* yGrad,
                            float* xGrad, size_t size)
{
    for (size_t i = 0; i < size; ++i)
        xGrad[i] = yGrad[i] * activationGradient<act>(x[i], y[i]);
}

}  // namespace

#define SINGLE_ARGS(...) __VA_ARGS__

#define CASE_ACTIVATION(ACTIVATION, STMTS)         \
    case ACTIVATION:                               \
    {                                              \
        constexpr Activation __ACT__ = ACTIVATION; \
        STMTS;                                     \
        break;                                     \
    }

#define SWITCH_ACTIVATION(activation, STMTS)                         \
    switch (activation)                                              \
    {                                                                \
        CASE_ACTIVATION(Activation::kRELU, SINGLE_ARGS(STMTS))       \
        CASE_ACTIVATION(Activation::kABS, SINGLE_ARGS(STMTS))        \
        CASE_ACTIVATION(Activation::kLOG, SINGLE_ARGS(STMTS))        \
        CASE_ACTIVATION(Activation::kNEG, SINGLE_ARGS(STMTS))        \
        CASE_ACTIVATION(Activation::kEXP, SINGLE_ARGS(STMTS))        \
        CASE_ACTIVATION(Activation::kSQUARE, SINGLE_ARGS(STMTS))     \
        CASE_ACTIVATION(Activation::kSIGMOID, SINGLE_ARGS(STMTS))    \
        CASE_ACTIVATION(Activation::kRECIPROCAL, SINGLE_ARGS(STMTS)) \
        CASE_ACTIVATION(Activation::kTANH, SINGLE_ARGS(STMTS))       \
        CASE_ACTIVATION(Activation::kSQRT, SINGLE_ARGS(STMTS))       \
        CASE_ACTIVATION(Activation::kLEAKY_RELU, SINGLE_ARGS(STMTS)) \
        CASE_ACTIVATION(Activation::kRELU_6, SINGLE_ARGS(STMTS))     \
        CASE_ACTIVATION(Activation::kELU, SINGLE_ARGS(STMTS))        \
        CASE_ACTIVATION(Activation::kSOFTPLUS, SINGLE_ARGS(STMTS))   \
        CASE_ACTIVATION(Activation::kSOFTSIGN, SINGLE_ARGS(STMTS))   \
    }

void runActivationHost(const float* x, float* y, size_t size, Activation op)
{
    SWITCH_ACTIVATION(op, activationHost<__ACT__>(x, y, size));
}

void runActivationGradientHost(const float* x, const float* y,
                               const float* yGrad, float* xGrad, size_t size,
                               Activation op)
{
    SWITCH_ACTIVATION(
        op, activationGradientHost<__ACT__>(x, y, yGrad, xGrad, size));
}

ActivationLayer::ActivationLayer(ID id, const Tensor::SPtr& t, Activation op)
    : DifferentiableLayer(id, {t},
                          {createTensor("", t->getShape(), t->getType())}),
      mOp(op)
{
}

void ActivationLayer::execute(const std::vector<float*>& inputs,
                              const std::vector<float*>& outputs,
                              const InputDict& /*inputDict*/)
{
    Tensor::SPtr tX = getInputs()[0];
    float* x = inputs[0];
    float* y = outputs[0];
    size_t size = tX->getCount();

    if (tX->getType() == MemoryType::kHOST_MEMORY)
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

void ActivationGradientLayer::execute(const std::vector<float*>& inputs,
                                      const std::vector<float*>& outputs,
                                      const InputDict& /*inputDict*/)
{
    Tensor::SPtr tX = getInputs()[0];
    float* x = inputs[0];
    float* y = inputs[1];
    float* yGrad = inputs[2];
    float* xGrad = outputs[0];
    size_t size = tX->getCount();

    if (tX->getType() == MemoryType::kHOST_MEMORY)
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

#define DEFINE_ACTIVATION_TENSOR(ACTIVATION, NAME)                             \
    Tensor::SPtr NAME(Tensor::SPtr t)                                          \
    {                                                                          \
        return createActivation(std::move(t), layers::Activation::ACTIVATION); \
    }

DEFINE_ACTIVATION_TENSOR(kRELU, relu);
DEFINE_ACTIVATION_TENSOR(kSIGMOID, sigmoid);
DEFINE_ACTIVATION_TENSOR(kTANH, tanh);
DEFINE_ACTIVATION_TENSOR(kSQUARE, square);
DEFINE_ACTIVATION_TENSOR(kABS, abs);
DEFINE_ACTIVATION_TENSOR(kNEG, neg);
DEFINE_ACTIVATION_TENSOR(kRECIPROCAL, reciprocal);
DEFINE_ACTIVATION_TENSOR(kLOG, log);
DEFINE_ACTIVATION_TENSOR(kSQRT, sqrt);
DEFINE_ACTIVATION_TENSOR(kEXP, exp);
DEFINE_ACTIVATION_TENSOR(kLEAKY_RELU, leaky_relu);
DEFINE_ACTIVATION_TENSOR(kRELU_6, relu6);
DEFINE_ACTIVATION_TENSOR(kELU, elu);
DEFINE_ACTIVATION_TENSOR(kSOFTPLUS, softplus);
DEFINE_ACTIVATION_TENSOR(kSOFTSIGN, softsign);

#undef DEFINE_ACTIVATION_TENSOR

}  // namespace core

#define DEFINE_ACTIVATION_ITENSOR(NAME)                             \
    ITensorPtr NAME(const ITensorPtr& t)                            \
    {                                                               \
        core::AbstractTensor::Ptr tensor = core::castITensorPtr(t); \
        return makeAbstractTensor(core::NAME(tensor->get()));       \
    }

DEFINE_ACTIVATION_ITENSOR(relu);
DEFINE_ACTIVATION_ITENSOR(sigmoid);
DEFINE_ACTIVATION_ITENSOR(tanh);
DEFINE_ACTIVATION_ITENSOR(square);
DEFINE_ACTIVATION_ITENSOR(abs);
DEFINE_ACTIVATION_ITENSOR(neg);
DEFINE_ACTIVATION_ITENSOR(reciprocal);
DEFINE_ACTIVATION_ITENSOR(log);
DEFINE_ACTIVATION_ITENSOR(sqrt);
DEFINE_ACTIVATION_ITENSOR(exp);
DEFINE_ACTIVATION_ITENSOR(leaky_relu);
DEFINE_ACTIVATION_ITENSOR(relu6);
DEFINE_ACTIVATION_ITENSOR(elu);
DEFINE_ACTIVATION_ITENSOR(softplus);
DEFINE_ACTIVATION_ITENSOR(softsign);

#undef DEFINE_ACTIVATION_ITENSOR

}  // namespace graphdl
