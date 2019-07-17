#include "softmax.h"

#include "abstractTensor.h"
#include "activation.h"
#include "elementwise.h"
#include "graph.h"
#include "reduce.h"

#include <utility>

namespace graphdl
{
namespace core
{
namespace layers
{
namespace
{
int calcOutSize(const Tensor::SPtr& tensor, int numAxes)
{
    TensorShape shape = tensor->getShape();
    int x = 1;

    for (unsigned i = 0; i < shape.size() - numAxes; ++i) x *= shape[i];

    return x;
}

void runSoftmaxHost(const float* x, float* w, float* y, size_t outSize,
                    size_t reduceSize)
{
    runActivationHost(x, y, outSize * reduceSize, Activation::kEXP);
    runReduceBackHost(y, w, outSize, reduceSize, ReduceType::kSUM);

    for (size_t i = 0; i < outSize * reduceSize; ++i) y[i] /= w[i / reduceSize];
}

void runSoftmaxGradientHost(const float* /* x */, const float* y,
                            const float* yGrad, float* xGrad, size_t outSize,
                            size_t reduceSize)
{
    for (unsigned i = 0; i < outSize; ++i)
    {
        float S = 0;
        for (unsigned j = 0; j < reduceSize; ++j)
            S += y[i * reduceSize + j] * yGrad[i * reduceSize + j];

        for (unsigned j = 0; j < reduceSize; ++j)
        {
            unsigned pos = i * reduceSize + j;
            xGrad[pos] = y[pos] * (yGrad[pos] - S);
        }
    }
}

}  // namespace

SoftmaxLayer::SoftmaxLayer(ID id, const Tensor::SPtr& x, int numAxes)
    : DifferentiableLayer(id, {x},
                          {createTensor("", x->getShape(), x->getType())}),
      mNumAxes(numAxes),
      mWorkingSpace(x->getType(), calcOutSize(x, numAxes))
{
    TensorShape shape = x->getShape();

    mOutSize = 1;
    for (unsigned i = 0; i < shape.size() - numAxes; ++i) mOutSize *= shape[i];

    mReduceSize = 1;
    for (unsigned i = shape.size() - numAxes; i < shape.size(); ++i)
        mReduceSize *= shape[i];
}

Layer::TensorMap SoftmaxLayer::gradients(Tensor::SPtr out, Tensor::SPtr outGrad)
{
    Tensor::SPtr in = mInputs[0].lock();
    Layer::SPtr layer =
        createLayer<SoftmaxGradientLayer>(in, mNumAxes, out, outGrad);
    return {{in, layer->getOutputs()[0]}};
}

void SoftmaxLayer::initialize()
{
    mWorkingSpace.allocate();
}

void SoftmaxLayer::execute(const std::vector<float*>& inputs,
                           const std::vector<float*>& outputs,
                           const InputDict& /*inputDict*/)
{
    float* x = inputs[0];
    float* w = mWorkingSpace.getValues();
    float* y = outputs[0];

    if (mInputs[0].lock()->getType() == MemoryType::kHOST_MEMORY)
        runSoftmaxHost(x, w, y, mOutSize, mReduceSize);
#ifdef CUDA_AVAILABLE
    else
        cuda::runSoftmaxDevice(x, w, y, mOutSize, mReduceSize);
#endif
}

SoftmaxLayer::~SoftmaxLayer()
{
    mWorkingSpace.free();
}

SoftmaxGradientLayer::SoftmaxGradientLayer(ID id, const Tensor::SPtr& x,
                                           int numAxes, Tensor::SPtr y,
                                           Tensor::SPtr yGrad)
    : Layer(id, {x, std::move(y), std::move(yGrad)},
            {createTensor("", x->getShape(), x->getType())}),
      mNumAxes(numAxes)
{
    TensorShape shape = x->getShape();

    mOutSize = 1;
    for (unsigned i = 0; i < shape.size() - numAxes; ++i) mOutSize *= shape[i];

    mReduceSize = 1;
    for (unsigned i = shape.size() - numAxes; i < shape.size(); ++i)
        mReduceSize *= shape[i];
}

void SoftmaxGradientLayer::execute(const std::vector<float*>& inputs,
                                   const std::vector<float*>& outputs,
                                   const InputDict& /*inputDict*/)
{
    float* x = inputs[0];
    float* y = inputs[1];
    float* yGrad = inputs[2];
    float* xGrad = outputs[0];

    if (mInputs[0].lock()->getType() == MemoryType::kHOST_MEMORY)
        runSoftmaxGradientHost(x, y, yGrad, xGrad, mOutSize, mReduceSize);
#ifdef CUDA_AVAILABLE
    else
        cuda::runSoftmaxGradientDevice(x, y, yGrad, xGrad, mOutSize,
                                       mReduceSize);
#endif
}

}  // namespace layers

Tensor::SPtr softmax(const Tensor::SPtr& tensor, int numAxes)
{
    if (numAxes <= 0) numAxes = tensor->getShape().size();

    Layer::SPtr layer = createLayer<layers::SoftmaxLayer>(tensor, numAxes);
    return layer->getOutputs()[0];
}

Tensor::SPtr softmax_c(const Tensor::SPtr& tensor, int numAxes)
{
    if (numAxes <= 0) numAxes = tensor->getShape().size();

    Tensor::SPtr max = reduceBack(tensor, numAxes, layers::ReduceType::kMAX);
    return softmax(tensor - max, numAxes);
}

Tensor::SPtr softmax_cross_entropy_with_logits(const Tensor::SPtr& logits,
                                               const Tensor::SPtr& labels)
{
    return log(reduceBack(exp(logits), 1, layers::ReduceType::kSUM)) -
           reduceBack(labels * logits, 1, layers::ReduceType::kSUM);
}

}  // namespace core

ITensorPtr softmax(const ITensorPtr& tensor, int numAxes)
{
    core::Tensor::SPtr t = core::castITensorPtr(tensor)->get();
    return core::makeAbstractTensor(core::softmax(t, numAxes));
}

ITensorPtr softmax_c(const ITensorPtr& tensor, int numAxes)
{
    core::Tensor::SPtr t = core::castITensorPtr(tensor)->get();
    return core::makeAbstractTensor(core::softmax_c(t, numAxes));
}

ITensorPtr softmax_cross_entropy_with_logits(const ITensorPtr& logits,
                                             const ITensorPtr& labels)
{
    core::Tensor::SPtr logits_ = core::castITensorPtr(logits)->get();
    core::Tensor::SPtr labels_ = core::castITensorPtr(labels)->get();
    return core::makeAbstractTensor(
        core::softmax_cross_entropy_with_logits(logits_, labels_));
}

}  // namespace graphdl
