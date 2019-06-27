#include "reduce.h"

#include "abstractTensor.h"
#include "elementwise.h"
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

}  // namespace

// TODO(adajed): for now there is only one possible ReduceType
void runReduceBackHost(const float* in, float* out, size_t outSize,
                       size_t reduceSize, ReduceType /*reduceType*/)
{
    for (size_t posY = 0; posY < outSize; ++posY)
    {
        out[posY] = 0.;
        for (size_t i = 0; i < reduceSize; ++i) out[posY] += in[i];
        in += reduceSize;
    }
}

// TODO(adajed): for now there is only one possible ReduceType
void runReduceBackGradientHost(const float* /*in*/, const float* /*out*/,
                               const float* outGrad, float* inGrad,
                               size_t outSize, size_t reduceSize,
                               ReduceType /*reduceType*/)
{
    for (size_t posY = 0; posY < outSize; ++posY)
    {
        for (size_t i = 0; i < reduceSize; ++i) inGrad[i] = outGrad[posY];
        inGrad += reduceSize;
    }
}

// TODO(adajed): for now there is only one possible ReduceType
void runReduceFrontHost(const float* in, float* out, size_t outSize,
                        size_t reduceSize, ReduceType /*reduceType*/)
{
    for (size_t posY = 0; posY < outSize; ++posY)
    {
        out[posY] = 0.;
        for (size_t i = 0; i < reduceSize; ++i)
            out[posY] += in[i * outSize + posY];
    }
}

// TODO(adajed): for now there is only one possible ReduceType
void runReduceFrontGradientHost(const float* /*in*/, const float* /*out*/,
                                const float* outGrad, float* inGrad,
                                size_t outSize, size_t reduceSize,
                                ReduceType /*reduceType*/)
{
    for (size_t posY = 0; posY < outSize; ++posY)
        for (size_t i = 0; i < reduceSize; ++i)
            inGrad[i * outSize + posY] = outGrad[posY];
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

}  // namespace core

ITensorPtr reduceSum(const ITensorPtr& t, int numAxes)
{
    core::AbstractTensor::Ptr tensor = core::castITensorPtr(t);
    return makeAbstractTensor(core::reduceBack(tensor->get(), numAxes,
                                               core::layers::ReduceType::kSUM));
}

}  // namespace graphdl
