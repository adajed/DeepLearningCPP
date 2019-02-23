#include "elementwise.h"

#include "abstractTensor.h"
#include "constant.h"
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
TensorShape getBigger(const TensorShape& s1, const TensorShape& s2)
{
    return s1.size() > s2.size() ? s1 : s2;
}

std::vector<Tensor::SPtr> createOutputs(const Tensor::SPtr& t1,
                                        const Tensor::SPtr& t2)
{
    assert(t1->getType() == t2->getType());
    TensorShape shape = getBigger(t1->getShape(), t2->getShape());
    return {createTensor("", shape, t1->getType())};
}

std::vector<Tensor::SPtr> createGradientOutputs(const Tensor::SPtr& t1,
                                                const Tensor::SPtr& t2)
{
    assert(t1->getType() == t2->getType());
    return {createTensor("", t1->getShape(), t1->getType()),
            createTensor("", t2->getShape(), t2->getType())};
}

}  // namespace

ElementwiseBackLayer::ElementwiseBackLayer(ID id, const Tensor::SPtr& t1,
                                   const Tensor::SPtr& t2, Elementwise op)
    : DifferentiableLayer(id, {t1, t2}, createOutputs(t1, t2)), mOp(op)
{
}

void ElementwiseBackLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr x1Tensor = mInputs[0].lock();
    Tensor::SPtr x2Tensor = mInputs[1].lock();
    x1Tensor->eval(inputs);
    x2Tensor->eval(inputs);

    float* x1 = x1Tensor->getMemory().getValues();
    float* x2 = x2Tensor->getMemory().getValues();
    float* y = mOutputs[0]->getMemory().getValues();
    size_t size1 = x1Tensor->getMemory().getCount();
    size_t size2 = x2Tensor->getMemory().getCount();

    if (x1Tensor->getType() == MemoryType::kHOST_MEMORY)
        runElementwiseBackHost(x1, size1, x2, size2, y, mOp);
#ifdef CUDA_AVAILABLE
    else
        cuda::runElementwiseBackDevice(x1, size1, x2, size2, y, mOp);
#endif
}

Layer::TensorMap ElementwiseBackLayer::gradients(Tensor::SPtr output,
                                             Tensor::SPtr outputGrad)
{
    assert(output == mOutputs[0]);

    std::vector<Tensor::SPtr> inputs = getInputs();
    Layer::SPtr layer = createLayer<ElementwiseBackGradientLayer>(
        inputs[0], inputs[1], output, outputGrad, mOp);

    std::vector<Tensor::SPtr> grads = layer->getOutputs();
    return {{inputs[0], grads[0]}, {inputs[1], grads[1]}};
}

ElementwiseBackGradientLayer::ElementwiseBackGradientLayer(
    ID id, const Tensor::SPtr& t1, const Tensor::SPtr& t2, Tensor::SPtr out,
    Tensor::SPtr outGrad, Elementwise op)
    : Layer(id, {t1, t2, std::move(out), std::move(outGrad)},
            createGradientOutputs(t1, t2)),
      mOp(op){};

void ElementwiseBackGradientLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr x1Tensor = mInputs[0].lock();
    Tensor::SPtr x2Tensor = mInputs[1].lock();
    Tensor::SPtr yGradTensor = mInputs[3].lock();
    x1Tensor->eval(inputs);
    x2Tensor->eval(inputs);
    yGradTensor->eval(inputs);

    float* x1 = x1Tensor->getMemory().getValues();
    float* x2 = x2Tensor->getMemory().getValues();
    float* yGrad = yGradTensor->getMemory().getValues();
    float* x1Grad = mOutputs[0]->getMemory().getValues();
    float* x2Grad = mOutputs[1]->getMemory().getValues();
    size_t size1 = x1Tensor->getMemory().getCount();
    size_t size2 = x2Tensor->getMemory().getCount();

    if (x1Tensor->getType() == MemoryType::kHOST_MEMORY)
        runElementwiseBackGradientHost(x1, size1, x2, size2, yGrad, x1Grad,
                                   x2Grad, mOp);
#ifdef CUDA_AVAILABLE
    else
        cuda::runElementwiseBackGradientDevice(x1, size1, x2, size2, yGrad,
                                           x1Grad, x2Grad, mOp);
#endif
}

ElementwiseFrontLayer::ElementwiseFrontLayer(
        ID id, const Tensor::SPtr& t1, const Tensor::SPtr& t2, Elementwise op)
    : DifferentiableLayer(id, {t1, t2}, createOutputs(t1, t2)), mOp(op)
{
}

void ElementwiseFrontLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr x1Tensor = mInputs[0].lock();
    Tensor::SPtr x2Tensor = mInputs[1].lock();
    x1Tensor->eval(inputs);
    x2Tensor->eval(inputs);

    float* x1 = x1Tensor->getMemory().getValues();
    float* x2 = x2Tensor->getMemory().getValues();
    float* y = mOutputs[0]->getMemory().getValues();
    size_t size1 = x1Tensor->getMemory().getCount();
    size_t size2 = x2Tensor->getMemory().getCount();

    if (x1Tensor->getType() == MemoryType::kHOST_MEMORY)
        runElementwiseFrontHost(x1, size1, x2, size2, y, mOp);
#ifdef CUDA_AVAILABLE
    else
        cuda::runElementwiseFrontDevice(x1, size1, x2, size2, y, mOp);
#endif
}

Layer::TensorMap ElementwiseFrontLayer::gradients(Tensor::SPtr output,
                                             Tensor::SPtr outputGrad)
{
    assert(output == mOutputs[0]);

    std::vector<Tensor::SPtr> inputs = getInputs();
    Layer::SPtr layer = createLayer<ElementwiseFrontGradientLayer>(
        inputs[0], inputs[1], output, outputGrad, mOp);

    std::vector<Tensor::SPtr> grads = layer->getOutputs();
    return {{inputs[0], grads[0]}, {inputs[1], grads[1]}};
}

ElementwiseFrontGradientLayer::ElementwiseFrontGradientLayer(
    ID id, const Tensor::SPtr& t1, const Tensor::SPtr& t2, Tensor::SPtr out,
    Tensor::SPtr outGrad, Elementwise op)
    : Layer(id, {t1, t2, std::move(out), std::move(outGrad)},
            createGradientOutputs(t1, t2)),
      mOp(op){};

void ElementwiseFrontGradientLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr x1Tensor = mInputs[0].lock();
    Tensor::SPtr x2Tensor = mInputs[1].lock();
    Tensor::SPtr yGradTensor = mInputs[3].lock();
    x1Tensor->eval(inputs);
    x2Tensor->eval(inputs);
    yGradTensor->eval(inputs);

    float* x1 = x1Tensor->getMemory().getValues();
    float* x2 = x2Tensor->getMemory().getValues();
    float* yGrad = yGradTensor->getMemory().getValues();
    float* x1Grad = mOutputs[0]->getMemory().getValues();
    float* x2Grad = mOutputs[1]->getMemory().getValues();
    size_t size1 = x1Tensor->getMemory().getCount();
    size_t size2 = x2Tensor->getMemory().getCount();

    if (x1Tensor->getType() == MemoryType::kHOST_MEMORY)
        runElementwiseFrontGradientHost(x1, size1, x2, size2, yGrad, x1Grad,
                                   x2Grad, mOp);
#ifdef CUDA_AVAILABLE
    else
        cuda::runElementwiseFrontGradientDevice(x1, size1, x2, size2, yGrad,
                                           x1Grad, x2Grad, mOp);
#endif
}

}  // namespace layers
namespace
{
TensorShape shorterShape(const TensorShape& s1, const TensorShape& s2)
{
    return s1.size() < s2.size() ? s1 : s2;
}

TensorShape longerShape(const TensorShape& s1, const TensorShape& s2)
{
    return s1.size() >= s2.size() ? s1 : s2;
}

bool checkShapesCompatibleBack(const TensorShape& s1, const TensorShape& s2)
{
    TensorShape shortShape = shorterShape(s1, s2);
    TensorShape longShape = longerShape(s1, s2);
    int sizeShort = shortShape.size();
    int sizeLong = longShape.size();

    for (int i = 0; i < sizeShort; ++i)
        if (shortShape[sizeShort - i - 1] != longShape[sizeLong - i - 1])
            return false;

    return true;
}

bool checkShapesCompatibleFront(const TensorShape& s1, const TensorShape& s2)
{
    TensorShape shortShape = shorterShape(s1, s2);
    TensorShape longShape = longerShape(s1, s2);
    int sizeShort = shortShape.size();
    int sizeLong = longShape.size();

    for (int i = 0; i < sizeShort; ++i)
        if (shortShape[i] != longShape[i])
            return false;

    return true;
}

}  // namespace

Tensor::SPtr elementwiseBack(const Tensor::SPtr& t1, const Tensor::SPtr& t2,
                               layers::Elementwise op)
{
    if (!checkShapesCompatibleBack(t1->getShape(), t2->getShape()))
        throw std::runtime_error("Shapes don\'t match");

    Layer::SPtr layer = createLayer<layers::ElementwiseBackLayer>(t1, t2, op);
    return layer->getOutputs()[0];
}

Tensor::SPtr elementwiseFront(const Tensor::SPtr& t1, const Tensor::SPtr& t2,
                              layers::Elementwise op)
{
    if (!checkShapesCompatibleFront(t1->getShape(), t2->getShape()))
        throw std::runtime_error("Shapes don\'t match");

    Layer::SPtr layer = createLayer<layers::ElementwiseFrontLayer>(t1, t2, op);
    return layer->getOutputs()[0];
}

Tensor::SPtr elementwise(const Tensor::SPtr& t1, const Tensor::SPtr& t2,
                              layers::Elementwise op)
{
    Layer::SPtr layer;
    if (checkShapesCompatibleBack(t1->getShape(), t2->getShape()))
        layer = createLayer<layers::ElementwiseBackLayer>(t1, t2, op);
    else if (checkShapesCompatibleFront(t1->getShape(), t2->getShape()))
        layer = createLayer<layers::ElementwiseFrontLayer>(t1, t2, op);
    else
        throw std::runtime_error("Shapes don\'t match");

    return layer->getOutputs()[0];
}


#define ELEMENTWISE_CORE(opName, op, elem)                              \
    Tensor::SPtr opName(const Tensor::SPtr& t1, const Tensor::SPtr& t2) \
    {                                                                   \
        return elementwise(t1, t2, layers::Elementwise::elem);    \
    }                                                                   \
    Tensor::SPtr opName(float val, const Tensor::SPtr& t2)              \
    {                                                                   \
        Tensor::SPtr t1 = constant(val, {}, t2->getType());             \
        return elementwise(t1, t2, layers::Elementwise::elem);    \
    }                                                                   \
    Tensor::SPtr opName(const Tensor::SPtr& t1, float val)              \
    {                                                                   \
        Tensor::SPtr t2 = constant(val, {}, t1->getType());             \
        return elementwise(t1, t2, layers::Elementwise::elem);    \
    }                                                                   \
    Tensor::SPtr op(const Tensor::SPtr& t1, const Tensor::SPtr& t2)     \
    {                                                                   \
        return opName(t1, t2);                                          \
    }                                                                   \
    Tensor::SPtr op(float val, const Tensor::SPtr& t2)                  \
    {                                                                   \
        return opName(val, t2);                                         \
    }                                                                   \
    Tensor::SPtr op(const Tensor::SPtr& t1, float val)                  \
    {                                                                   \
        return opName(t1, val);                                         \
    }

ELEMENTWISE_CORE(add, operator+, kADD)
ELEMENTWISE_CORE(sub, operator-, kSUB)
ELEMENTWISE_CORE(mul, operator*, kMUL)
ELEMENTWISE_CORE(div, operator/, kDIV)

}  // namespace core

#define ELEMENTWISE(opName, op)                                                \
    ITensorPtr opName(const ITensorPtr& t1, const ITensorPtr& t2)              \
    {                                                                          \
        core::AbstractTensor::Ptr tensor1 = core::castITensorPtr(t1);          \
        core::AbstractTensor::Ptr tensor2 = core::castITensorPtr(t2);          \
        return makeAbstractTensor(                                             \
            core::opName(tensor1->get(), tensor2->get()));                     \
    }                                                                          \
    ITensorPtr opName(float val, const ITensorPtr& t2)                         \
    {                                                                          \
        core::AbstractTensor::Ptr tensor2 = core::castITensorPtr(t2);          \
        return makeAbstractTensor(core::opName(val, tensor2->get()));          \
    }                                                                          \
    ITensorPtr opName(const ITensorPtr& t1, float val)                         \
    {                                                                          \
        core::AbstractTensor::Ptr tensor1 = core::castITensorPtr(t1);          \
        return makeAbstractTensor(core::opName(tensor1->get(), val));          \
    }                                                                          \
                                                                               \
    ITensorPtr op(const ITensorPtr& t1, const ITensorPtr& t2)                  \
    {                                                                          \
        return opName(t1, t2);                                                 \
    }                                                                          \
    ITensorPtr op(float val, const ITensorPtr& t2) { return opName(val, t2); } \
    ITensorPtr op(const ITensorPtr& t1, float val) { return opName(t1, val); }

ELEMENTWISE(add, operator+)
ELEMENTWISE(sub, operator-)
ELEMENTWISE(mul, operator*)
ELEMENTWISE(div, operator/)

}  // namespace graphdl
