#include "elementwise.h"

#include "abstractTensor.h"
#include "graph.h"
#include "graphdl_ops.h"

#include <cassert>
#include <utility>

namespace graphdl
{
namespace core
{
namespace layers
{
namespace
{
template <Elementwise elem>
float op(float f1, float f2);
template <>
float op<Elementwise::kADD>(float f1, float f2)
{
    return f1 + f2;
}
template <>
float op<Elementwise::kSUB>(float f1, float f2)
{
    return f1 - f2;
}
template <>
float op<Elementwise::kMUL>(float f1, float f2)
{
    return f1 * f2;
}
template <>
float op<Elementwise::kDIV>(float f1, float f2)
{
    return f1 / f2;
}

template <Elementwise elem>
void elementwise(std::size_t size, float* x1, float* x2, float* y)
{
    for (std::size_t pos = 0; pos < size; ++pos)
        y[pos] = op<elem>(x1[pos], x2[pos]);
}

void runElementwiseHost(std::size_t size, float* x1, float* x2, float* y,
                        Elementwise op)
{
    switch (op)
    {
    case Elementwise::kADD:
        elementwise<Elementwise::kADD>(size, x1, x2, y);
        return;
    case Elementwise::kSUB:
        elementwise<Elementwise::kSUB>(size, x1, x2, y);
        return;
    case Elementwise::kMUL:
        elementwise<Elementwise::kMUL>(size, x1, x2, y);
        return;
    case Elementwise::kDIV:
        elementwise<Elementwise::kDIV>(size, x1, x2, y);
        return;
    }
}

template <Elementwise elem>
float opGrad1(float f1, float f2);
template <>
float opGrad1<Elementwise::kADD>(float /* f1 */, float /* f2 */)
{
    return 1.;
}
template <>
float opGrad1<Elementwise::kSUB>(float /* f1 */, float /* f2 */)
{
    return 1.;
}
template <>
float opGrad1<Elementwise::kMUL>(float /* f1 */, float f2)
{
    return f2;
}
template <>
float opGrad1<Elementwise::kDIV>(float /* f1 */, float f2)
{
    return 1. / f2;
}

template <Elementwise elem>
float opGrad2(float f1, float f2);
template <>
float opGrad2<Elementwise::kADD>(float /* f1 */, float /* f2 */)
{
    return 1.;
}
template <>
float opGrad2<Elementwise::kSUB>(float /* f1 */, float /* f2 */)
{
    return -1.;
}
template <>
float opGrad2<Elementwise::kMUL>(float f1, float /* f2 */)
{
    return f1;
}
template <>
float opGrad2<Elementwise::kDIV>(float f1, float f2)
{
    return -f1 / (f2 * f2);
}

template <Elementwise elem>
void elementwiseGradient(std::size_t size, float* x1, float* x2, float* yG,
                         float* x1G, float* x2G)
{
    for (std::size_t pos = 0; pos < size; ++pos)
    {
        x1G[pos] = yG[pos] * opGrad1<elem>(x1[pos], x2[pos]);
        x2G[pos] = yG[pos] * opGrad2<elem>(x1[pos], x2[pos]);
    }
}

void runElementwiseGradientHost(std::size_t size, float* x1, float* x2,
                                float* yG, float* x1G, float* x2G,
                                Elementwise op)
{
    switch (op)
    {
    case Elementwise::kADD:
        elementwiseGradient<Elementwise::kADD>(size, x1, x2, yG, x1G, x2G);
        return;
    case Elementwise::kSUB:
        elementwiseGradient<Elementwise::kSUB>(size, x1, x2, yG, x1G, x2G);
        return;
    case Elementwise::kMUL:
        elementwiseGradient<Elementwise::kMUL>(size, x1, x2, yG, x1G, x2G);
        return;
    case Elementwise::kDIV:
        elementwiseGradient<Elementwise::kDIV>(size, x1, x2, yG, x1G, x2G);
        return;
    }
}

std::vector<Tensor::SPtr> createOutputs(const Tensor::SPtr& t1,
                                        const Tensor::SPtr& t2)
{
    assert(t1->getType() == t2->getType());
    return {createTensor("", t1->getShape(), t1->getType())};
}

std::vector<Tensor::SPtr> createGradientOutputs(const Tensor::SPtr& t1,
                                                const Tensor::SPtr& t2)
{
    assert(t1->getType() == t2->getType());
    return {createTensor("", t1->getShape(), t1->getType()),
            createTensor("", t2->getShape(), t2->getType())};
}

}  // namespace

ElementwiseLayer::ElementwiseLayer(ID id, const Tensor::SPtr& t1,
                                   const Tensor::SPtr& t2, Elementwise op)
    : DifferentiableLayer(id, {t1, t2}, createOutputs(t1, t2)), mOp(op)
{
    assert(t1->getShape() == t2->getShape());
}

void ElementwiseLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr i0 = mInputs[0].lock();
    Tensor::SPtr i1 = mInputs[1].lock();

    i0->eval(inputs);
    i1->eval(inputs);

    float* input0 = i0->getMemory().getValues();
    float* input1 = i1->getMemory().getValues();
    float* output = mOutputs[0]->getMemory().getValues();
    std::size_t size = mOutputs[0]->getMemory().getCount();

    if (i0->getType() == MemoryType::kHOST_MEMORY)
        runElementwiseHost(size, input0, input1, output, mOp);
#ifdef CUDA_AVAILABLE
    else
        cuda::runElementwiseDevice(size, input0, input1, output, mOp);
#endif
}

Layer::TensorMap ElementwiseLayer::gradients(Tensor::SPtr output,
                                             Tensor::SPtr outputGrad)
{
    assert(output == mOutputs[0]);

    std::vector<Tensor::SPtr> inputs = getInputs();
    Layer::SPtr layer = createLayer<ElementwiseGradientLayer>(
        inputs[0], inputs[1], output, outputGrad, mOp);

    std::vector<Tensor::SPtr> grads = layer->getOutputs();
    return {{inputs[0], grads[0]}, {inputs[1], grads[1]}};
}

ElementwiseGradientLayer::ElementwiseGradientLayer(
    ID id, const Tensor::SPtr& t1, const Tensor::SPtr& t2, Tensor::SPtr out,
    Tensor::SPtr outGrad, Elementwise op)
    : Layer(id, {t1, t2, std::move(out), std::move(outGrad)},
            createGradientOutputs(t1, t2)),
      mOp(op){};

void ElementwiseGradientLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr input1 = mInputs[0].lock();
    Tensor::SPtr input2 = mInputs[1].lock();
    Tensor::SPtr outputGrad = mInputs[3].lock();
    input1->eval(inputs);
    input2->eval(inputs);
    outputGrad->eval(inputs);

    float* in1 = input1->getMemory().getValues();
    float* in2 = input2->getMemory().getValues();
    float* outGrad = outputGrad->getMemory().getValues();
    float* gradient1 = mOutputs[0]->getMemory().getValues();
    float* gradient2 = mOutputs[1]->getMemory().getValues();
    std::size_t size = input1->getMemory().getCount();

    if (input1->getType() == MemoryType::kHOST_MEMORY)
        runElementwiseGradientHost(size, in1, in2, outGrad, gradient1,
                                   gradient2, mOp);
#ifdef CUDA_AVAILABLE
    else
        cuda::runElementwiseGradientDevice(size, in1, in2, outGrad, gradient1,
                                           gradient2, mOp);
#endif
}

}  // namespace layers

Tensor::SPtr createElementwise(const Tensor::SPtr& t1, const Tensor::SPtr& t2,
                               layers::Elementwise op)
{
    if (t1->getShape() != t2->getShape())
        throw std::runtime_error("Shapes don\'t match");

    Layer::SPtr layer = createLayer<layers::ElementwiseLayer>(t1, t2, op);
    return layer->getOutputs()[0];
}

Tensor::SPtr add(const Tensor::SPtr& t1, const Tensor::SPtr& t2)
{
    return createElementwise(t1, t2, layers::Elementwise::kADD);
}

Tensor::SPtr sub(const Tensor::SPtr& t1, const Tensor::SPtr& t2)
{
    return createElementwise(t1, t2, layers::Elementwise::kSUB);
}

Tensor::SPtr mul(const Tensor::SPtr& t1, const Tensor::SPtr& t2)
{
    return createElementwise(t1, t2, layers::Elementwise::kMUL);
}

Tensor::SPtr div(const Tensor::SPtr& t1, const Tensor::SPtr& t2)
{
    return createElementwise(t1, t2, layers::Elementwise::kDIV);
}

Tensor::SPtr operator+(const Tensor::SPtr& t1, const Tensor::SPtr& t2)
{
    return add(t1, t2);
}

Tensor::SPtr operator-(const Tensor::SPtr& t1, const Tensor::SPtr& t2)
{
    return sub(t1, t2);
}

Tensor::SPtr operator*(const Tensor::SPtr& t1, const Tensor::SPtr& t2)
{
    return mul(t1, t2);
}

Tensor::SPtr operator/(const Tensor::SPtr& t1, const Tensor::SPtr& t2)
{
    return div(t1, t2);
}

}  // namespace core

ITensorPtr add(const ITensorPtr& t1, const ITensorPtr& t2)
{
    core::AbstractTensor::Ptr tensor1 = core::castITensorPtr(t1);
    core::AbstractTensor::Ptr tensor2 = core::castITensorPtr(t2);
    return makeAbstractTensor(core::add(tensor1->get(), tensor2->get()));
}

ITensorPtr sub(const ITensorPtr& t1, const ITensorPtr& t2)
{
    core::AbstractTensor::Ptr tensor1 = core::castITensorPtr(t1);
    core::AbstractTensor::Ptr tensor2 = core::castITensorPtr(t2);
    return makeAbstractTensor(core::sub(tensor1->get(), tensor2->get()));
}

ITensorPtr mul(const ITensorPtr& t1, const ITensorPtr& t2)
{
    core::AbstractTensor::Ptr tensor1 = core::castITensorPtr(t1);
    core::AbstractTensor::Ptr tensor2 = core::castITensorPtr(t2);
    return makeAbstractTensor(core::mul(tensor1->get(), tensor2->get()));
}

ITensorPtr div(const ITensorPtr& t1, const ITensorPtr& t2)
{
    core::AbstractTensor::Ptr tensor1 = core::castITensorPtr(t1);
    core::AbstractTensor::Ptr tensor2 = core::castITensorPtr(t2);
    return makeAbstractTensor(core::div(tensor1->get(), tensor2->get()));
}

ITensorPtr operator+(const ITensorPtr& t1, const ITensorPtr& t2)
{
    return add(t1, t2);
}

ITensorPtr operator-(const ITensorPtr& t1, const ITensorPtr& t2)
{
    return sub(t1, t2);
}

ITensorPtr operator*(const ITensorPtr& t1, const ITensorPtr& t2)
{
    return mul(t1, t2);
}

ITensorPtr operator/(const ITensorPtr& t1, const ITensorPtr& t2)
{
    return div(t1, t2);
}

}  // namespace graphdl
