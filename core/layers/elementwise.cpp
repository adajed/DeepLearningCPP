#include "elementwise.h"
#include "abstractTensor.h"

#include <cassert>
#include <utility>
#include "graph.h"
#include "graphdl_ops.h"

namespace graphdl
{
namespace core
{
namespace layers
{
namespace
{
std::vector<Tensor::SPtr> createOutputs(const Tensor::SPtr& t1)
{
    return {createTensor("", t1->getShape())};
}

std::vector<Tensor::SPtr> createGradientOutputs(const Tensor::SPtr& t1,
                                                const Tensor::SPtr& t2)
{
    return {createTensor("", t1->getShape()), createTensor("", t2->getShape())};
}

}  // namespace

ElementwiseLayer::ElementwiseLayer(ID id, const Tensor::SPtr& t1,
                                   const Tensor::SPtr& t2, Elementwise op)
    : DifferentiableLayer(id, {t1, t2}, createOutputs(t1)), mOp(op)
{
    assert(t1->getShape() == t2->getShape());
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

void ElementwiseLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr i0 = mInputs[0].lock();
    Tensor::SPtr i1 = mInputs[1].lock();

    i0->eval(inputs);
    i1->eval(inputs);

    Memory input0 = i0->getMemory();
    Memory input1 = i1->getMemory();
    Memory output = mOutputs[0]->getMemory();

    for (std::size_t i = 0; i < output.getCount(); ++i)
        output[i] = mFun(input0[i], input1[i]);
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
      mOp(op)
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

void ElementwiseGradientLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr input1 = mInputs[0].lock();
    Tensor::SPtr input2 = mInputs[1].lock();
    Tensor::SPtr outputGrad = mInputs[3].lock();
    input1->eval(inputs);
    input2->eval(inputs);
    outputGrad->eval(inputs);

    Memory in1 = input1->getMemory();
    Memory in2 = input2->getMemory();
    Memory outGrad = outputGrad->getMemory();
    Memory gradient1 = mOutputs[0]->getMemory();
    Memory gradient2 = mOutputs[1]->getMemory();

    for (std::size_t i = 0; i < in1.getCount(); ++i)
    {
        gradient1[i] = outGrad[i] * mFun1(in1[i], in2[i]);
        gradient2[i] = outGrad[i] * mFun2(in1[i], in2[i]);
    }
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
