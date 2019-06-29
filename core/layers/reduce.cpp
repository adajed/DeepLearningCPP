#include "reduce.h"

#include "abstractTensor.h"
#include "elementwise.h"
#include "graph.h"
#include "graphdl_ops.h"

#include <cassert>
#include <cfloat>
#include <utility>

namespace graphdl
{
namespace core
{
namespace layers
{
namespace
{
Tensor::SPtr createOutputBack(const Tensor::SPtr& t, int numAxes)
{
    int n = t->getShape().size();
    TensorShape shape = t->getShape().subshape(0, n - numAxes);
    return createTensor("", shape, t->getType());
}

Tensor::SPtr createOutputFront(const Tensor::SPtr& t, int numAxes)
{
    int n = t->getShape().size();
    TensorShape shape = t->getShape().subshape(numAxes, n - numAxes);
    return createTensor("", shape, t->getType());
}

template <ReduceType op>
float initialValue();
template <>
float initialValue<ReduceType::kSUM>()
{
    return 0.;
}
template <>
float initialValue<ReduceType::kMIN>()
{
    return FLT_MAX;
}
template <>
float initialValue<ReduceType::kMAX>()
{
    return -FLT_MAX;
}

template <ReduceType op>
float reduce(float acc, float val);
template <>
float reduce<ReduceType::kSUM>(float acc, float val)
{
    return acc + val;
}
template <>
float reduce<ReduceType::kMAX>(float acc, float val)
{
    return acc > val ? acc : val;
}
template <>
float reduce<ReduceType::kMIN>(float acc, float val)
{
    return acc < val ? acc : val;
}

template <ReduceType op>
float reduceGrad(float x, float y);
template <>
float reduceGrad<ReduceType::kSUM>(float /* x */, float /* y */)
{
    return 1.;
}
template <>
float reduceGrad<ReduceType::kMAX>(float x, float y)
{
    return float(x == y);
}
template <>
float reduceGrad<ReduceType::kMIN>(float x, float y)
{
    return float(x == y);
}

template <ReduceType op>
void reduceBackHost(const float* x, float* y, size_t outSize, size_t reduceSize)
{
    for (size_t pos_y = 0; pos_y < outSize; ++pos_y)
    {
        float val = initialValue<op>();
        for (size_t i = 0; i < reduceSize; ++i) val = reduce<op>(val, x[i]);
        y[pos_y] = val;
        x += reduceSize;
    }
}

template <ReduceType op>
void reduceBackGradientHost(const float* in, const float* out,
                            const float* outGrad, float* inGrad, size_t outSize,
                            size_t reduceSize)
{
    for (size_t posY = 0; posY < outSize; ++posY)
    {
        for (size_t i = 0; i < reduceSize; ++i)
            inGrad[i] = outGrad[posY] * reduceGrad<op>(in[i], out[posY]);
        in += reduceSize;
        inGrad += reduceSize;
    }
}

template <ReduceType op>
void reduceFrontHost(const float* x, float* y, size_t outSize, size_t reduceSize)
{
    for (size_t posY = 0; posY < outSize; ++posY)
    {
        float val = initialValue<op>();
        for (size_t i = 0; i < reduceSize; ++i)
            val = reduce<op>(val, x[i * outSize + posY]);
        y[posY] = val;
    }
}

template <ReduceType op>
void reduceFrontGradientHost(const float* x, const float* y,
                             const float* yGrad, float* xGrad,
                             size_t outSize, size_t reduceSize)
{
    for (size_t posY = 0; posY < outSize; ++posY)
    {
        for (size_t i = 0; i < reduceSize; ++i)
            xGrad[i * outSize + posY] =
                yGrad[posY] * reduceGrad<op>(x[i * outSize + posY], y[posY]);
    }
}

}  // namespace

void runReduceBackHost(const float* in, float* out, size_t outSize,
                       size_t reduceSize, ReduceType reduceType)
{
    switch (reduceType)
    {
    case ReduceType::kSUM:
        reduceBackHost<ReduceType::kSUM>(in, out, outSize, reduceSize);
        break;
    case ReduceType::kMAX:
        reduceBackHost<ReduceType::kMAX>(in, out, outSize, reduceSize);
        break;
    case ReduceType::kMIN:
        reduceBackHost<ReduceType::kMIN>(in, out, outSize, reduceSize);
        break;
    }
}

void runReduceBackGradientHost(const float* in, const float* out,
                               const float* outGrad, float* inGrad,
                               size_t outSize, size_t reduceSize,
                               ReduceType reduceType)
{
    switch (reduceType)
    {
    case ReduceType::kSUM:
        reduceBackGradientHost<ReduceType::kSUM>(in, out, outGrad, inGrad,
                                                 outSize, reduceSize);
        break;
    case ReduceType::kMAX:
        reduceBackGradientHost<ReduceType::kMAX>(in, out, outGrad, inGrad,
                                                 outSize, reduceSize);
        break;
    case ReduceType::kMIN:
        reduceBackGradientHost<ReduceType::kMIN>(in, out, outGrad, inGrad,
                                                 outSize, reduceSize);
        break;
    }
}

void runReduceFrontHost(const float* in, float* out, size_t outSize,
                        size_t reduceSize, ReduceType reduceType)
{
    switch (reduceType)
    {
    case ReduceType::kSUM:
        reduceFrontHost<ReduceType::kSUM>(in, out, outSize, reduceSize);
        break;
    case ReduceType::kMAX:
        reduceFrontHost<ReduceType::kMAX>(in, out, outSize, reduceSize);
        break;
    case ReduceType::kMIN:
        reduceFrontHost<ReduceType::kMIN>(in, out, outSize, reduceSize);
        break;
    }
}

void runReduceFrontGradientHost(const float* in, const float* out,
                                const float* outGrad, float* inGrad,
                                size_t outSize, size_t reduceSize,
                                ReduceType reduceType)
{
    switch (reduceType)
    {
    case ReduceType::kSUM:
        reduceFrontGradientHost<ReduceType::kSUM>(
                in, out, outGrad, inGrad, outSize, reduceSize);
        break;
    case ReduceType::kMAX:
        reduceFrontGradientHost<ReduceType::kMAX>(
                in, out, outGrad, inGrad, outSize, reduceSize);
        break;
    case ReduceType::kMIN:
        reduceFrontGradientHost<ReduceType::kMIN>(
                in, out, outGrad, inGrad, outSize, reduceSize);
        break;
    }
}

ReduceBackLayer::ReduceBackLayer(ID id, const Tensor::SPtr& tensor, int numAxes,
                                 ReduceType reduceType)
    : DifferentiableLayer(id, {tensor}, {createOutputBack(tensor, numAxes)}),
      mNumAxes(numAxes),
      mReduceType(reduceType)
{
}

void ReduceBackLayer::execute(const std::vector<float*>& inputs,
                              const std::vector<float*>& outputs,
                              const InputDict& /*inputDict*/)
{
    Tensor::SPtr in = getInputs()[0];

    float* input = inputs[0];
    float* output = outputs[0];
    std::vector<int> shape = in->getShape();
    size_t outSize = 1, reduceSize = 1;
    for (unsigned i = 0; i < shape.size() - mNumAxes; ++i) outSize *= shape[i];
    for (unsigned i = shape.size() - mNumAxes; i < shape.size(); ++i)
        reduceSize *= shape[i];

    if (in->getType() == MemoryType::kHOST_MEMORY)
        runReduceBackHost(input, output, outSize, reduceSize, mReduceType);
#ifdef CUDA_AVAILABLE
    else
        cuda::runReduceBackDevice(input, output, outSize, reduceSize,
                                  mReduceType);
#endif
}

Layer::TensorMap ReduceBackLayer::gradients(Tensor::SPtr out,
                                            Tensor::SPtr outGrad)
{
    assert(out == mOutputs[0]);

    Tensor::SPtr in = getInputs()[0];
    Layer::SPtr layer = createLayer<ReduceBackGradientLayer>(
        in, out, outGrad, mNumAxes, mReduceType);
    return {{in, layer->getOutputs()[0]}};
}

ReduceBackGradientLayer::ReduceBackGradientLayer(ID id, const Tensor::SPtr& in,
                                                 const Tensor::SPtr& out,
                                                 const Tensor::SPtr& outGrad,
                                                 int numAxes,
                                                 ReduceType reduceType)
    : Layer(id, {in, out, outGrad},
            {createTensor("", in->getShape(), in->getType())}),
      mNumAxes(numAxes),
      mReduceType(reduceType)
{
}

void ReduceBackGradientLayer::execute(const std::vector<float*>& inputs,
                                      const std::vector<float*>& outputs,
                                      const InputDict& /*inputDict*/)
{
    Tensor::SPtr outputGrad = getInputs()[2];

    float* in = inputs[0];
    float* out = inputs[1];
    float* outGrad = inputs[2];
    float* inGrad = outputs[0];
    std::vector<int> shape = mOutputs[0]->getShape();
    size_t outSize = 1, reduceSize = 1;
    for (unsigned i = 0; i < shape.size() - mNumAxes; ++i) outSize *= shape[i];
    for (unsigned i = shape.size() - mNumAxes; i < shape.size(); ++i)
        reduceSize *= shape[i];

    if (outputGrad->getType() == MemoryType::kHOST_MEMORY)
        runReduceBackGradientHost(in, out, outGrad, inGrad, outSize, reduceSize,
                                  mReduceType);
#ifdef CUDA_AVAILABLE
    else
        cuda::runReduceBackGradientDevice(in, out, outGrad, inGrad, outSize,
                                          reduceSize, mReduceType);
#endif
}

ReduceFrontLayer::ReduceFrontLayer(ID id, const Tensor::SPtr& tensor,
                                   int numAxes, ReduceType reduceType)
    : DifferentiableLayer(id, {tensor}, {createOutputFront(tensor, numAxes)}),
      mNumAxes(numAxes),
      mReduceType(reduceType)
{
}

void ReduceFrontLayer::execute(const std::vector<float*>& inputs,
                               const std::vector<float*>& outputs,
                               const InputDict& /*inputDict*/)
{
    Tensor::SPtr input = getInputs()[0];

    float* in = inputs[0];
    float* out = outputs[0];
    std::vector<int> shape = input->getShape();
    size_t outSize = 1, reduceSize = 1;
    for (int i = 0; i < mNumAxes; ++i) reduceSize *= shape[i];
    for (int i = mNumAxes; i < shape.size(); ++i) outSize *= shape[i];

    if (input->getType() == MemoryType::kHOST_MEMORY)
        runReduceFrontHost(in, out, outSize, reduceSize, mReduceType);
#ifdef CUDA_AVAILABLE
    else
        cuda::runReduceFrontDevice(in, out, outSize, reduceSize, mReduceType);
#endif
}

Layer::TensorMap ReduceFrontLayer::gradients(Tensor::SPtr out,
                                             Tensor::SPtr outGrad)
{
    assert(out == mOutputs[0]);

    Tensor::SPtr in = getInputs()[0];
    Layer::SPtr layer = createLayer<ReduceFrontGradientLayer>(
        in, out, outGrad, mNumAxes, mReduceType);
    return {{in, layer->getOutputs()[0]}};
}

ReduceFrontGradientLayer::ReduceFrontGradientLayer(
    ID id, const Tensor::SPtr& in, const Tensor::SPtr& out,
    const Tensor::SPtr& outGrad, int numAxes, ReduceType reduceType)
    : Layer(id, {in, out, outGrad},
            {createTensor("", in->getShape(), in->getType())}),
      mNumAxes(numAxes),
      mReduceType(reduceType)
{
}

void ReduceFrontGradientLayer::execute(const std::vector<float*>& inputs,
                                       const std::vector<float*>& outputs,
                                       const InputDict& /*inputDict*/)
{
    Tensor::SPtr outputGrad = getInputs()[2];

    float* in = inputs[0];
    float* out = inputs[1];
    float* outGrad = inputs[2];
    float* inGrad = outputs[0];
    std::vector<int> shape = mOutputs[0]->getShape();
    size_t outSize = 1, reduceSize = 1;
    for (int i = 0; i < mNumAxes; ++i) reduceSize *= shape[i];
    for (int i = mNumAxes; i < shape.size(); ++i) outSize *= shape[i];

    if (outputGrad->getType() == MemoryType::kHOST_MEMORY)
        runReduceFrontGradientHost(in, out, outGrad, inGrad, outSize,
                                   reduceSize, mReduceType);
#ifdef CUDA_AVAILABLE
    else
        cuda::runReduceFrontGradientDevice(in, out, outGrad, inGrad, outSize,
                                           reduceSize, mReduceType);
#endif
}

}  // namespace layers

Tensor::SPtr reduceBack(const Tensor::SPtr& t, int numAxes,
                        layers::ReduceType reduceType)
{
    if (numAxes <= 0) numAxes = t->getShape().size();
    Layer::SPtr layer =
        createLayer<layers::ReduceBackLayer>(t, numAxes, reduceType);
    return layer->getOutputs()[0];
}

Tensor::SPtr reduceFront(const Tensor::SPtr& t, int numAxes,
                         layers::ReduceType reduceType)
{
    if (numAxes <= 0) numAxes = t->getShape().size();
    Layer::SPtr layer =
        createLayer<layers::ReduceFrontLayer>(t, numAxes, reduceType);
    return layer->getOutputs()[0];
}

Tensor::SPtr reduceMean(const Tensor::SPtr& t, int numAxes)
{
    if (numAxes <= 0) numAxes = t->getShape().size();
    size_t size = 1;
    for (int i = t->getShape().size() - numAxes; i < t->getShape().size(); ++i)
        size *= t->getShape()[i];

    Tensor::SPtr sum = reduceBack(t, numAxes, layers::ReduceType::kSUM);
    return sum * (1. / float(size));
}

}  // namespace core

template <core::layers::ReduceType reduceType>
ITensorPtr  reduceOperation(const ITensorPtr& t, int numAxes)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::reduceBack(tensor->get(), numAxes,
                                               reduceType));
}

ITensorPtr reduceSum(const ITensorPtr& t, int numAxes)
{
    return reduceOperation<core::layers::ReduceType::kSUM>(t, numAxes);
}

ITensorPtr reduceMax(const ITensorPtr& t, int numAxes)
{
    return reduceOperation<core::layers::ReduceType::kMAX>(t, numAxes);
}

ITensorPtr reduceMean(const ITensorPtr& t, int numAxes)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::reduceMean(tensor->get(), numAxes));
}

ITensorPtr reduceMin(const ITensorPtr& t, int numAxes)
{
    return reduceOperation<core::layers::ReduceType::kMIN>(t, numAxes);
}

}  // namespace graphdl
