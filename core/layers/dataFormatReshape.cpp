#include "dataFormatReshape.h"

#include "abstractTensor.h"
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
Tensor::SPtr createNchwTensor(const Tensor::SPtr& tensor)
{
    TensorShape shape = tensor->getShape();
    int temp = shape[3];

    shape[3] = shape[2];
    shape[2] = shape[1];
    shape[1] = temp;

    return createTensor("", shape, tensor->getType());
}

Tensor::SPtr createNhwcTensor(const Tensor::SPtr& tensor)
{
    TensorShape shape = tensor->getShape();
    int temp = shape[1];

    shape[1] = shape[2];
    shape[2] = shape[3];
    shape[3] = temp;

    return createTensor("", shape, tensor->getType());
}

}  // namespace

void runNhwc2NchwHost(const float* in, float* out, const int* outShape)
{
#define POS_IN(n, x, y, c) \
    ((((n)*outShape[2] + (x)) * outShape[3] + (y)) * outShape[1] + (c))

#define POS_OUT(n, c, x, y) \
    ((((n)*outShape[1] + (c)) * outShape[2] + (x)) * outShape[3] + (y))

    for (int n = 0; n < outShape[0]; ++n)
        for (int c = 0; c < outShape[1]; ++c)
            for (int x = 0; x < outShape[2]; ++x)
                for (int y = 0; y < outShape[3]; ++y)
                    out[POS_OUT(n, c, x, y)] = in[POS_IN(n, x, y, c)];

#undef POS_IN
#undef POS_OUT
}

void runNchw2NhwcHost(const float* in, float* out, const int* outShape)
{
#define POS_IN(n, c, x, y) \
    ((((n)*outShape[3] + (c)) * outShape[1] + (x)) * outShape[2] + (y))

#define POS_OUT(n, x, y, c) \
    ((((n)*outShape[1] + (x)) * outShape[2] + (y)) * outShape[3] + (c))

    for (int n = 0; n < outShape[0]; ++n)
        for (int x = 0; x < outShape[1]; ++x)
            for (int y = 0; y < outShape[2]; ++y)
                for (int c = 0; c < outShape[3]; ++c)
                    out[POS_OUT(n, x, y, c)] = in[POS_IN(n, c, x, y)];

#undef POS_IN
#undef POS_OUT
}

Nhwc2NchwLayer::Nhwc2NchwLayer(ID id, const Tensor::SPtr& tensor)
    : DifferentiableLayer(id, {tensor}, {createNchwTensor(tensor)})
{
}

Layer::TensorMap Nhwc2NchwLayer::gradients(Tensor::SPtr out,
                                           Tensor::SPtr outGrad)
{
    assert(out == mOutputs[0]);

    Layer::SPtr layer = createLayer<Nchw2NhwcLayer>(outGrad);

    return {{getInputs()[0], layer->getOutputs()[0]}};
}

void Nhwc2NchwLayer::execute(const std::vector<float*>& inputs,
                             const std::vector<float*>& outputs,
                             const InputDict& /*inputDict*/)
{
    float* in = inputs[0];
    float* out = outputs[0];
    std::vector<int> outShape = getOutputs()[0]->getShape();

    if (getInputs()[0]->getType() == MemoryType::kHOST_MEMORY)
        runNhwc2NchwHost(in, out, outShape.data());
#ifdef CUDA_AVAILABLE
    else
        cuda::runNhwc2NchwDevice(in, out, outShape.data());
#endif
}

Nchw2NhwcLayer::Nchw2NhwcLayer(ID id, const Tensor::SPtr& tensor)
    : DifferentiableLayer(id, {tensor}, {createNhwcTensor(tensor)})
{
}

Layer::TensorMap Nchw2NhwcLayer::gradients(Tensor::SPtr out,
                                           Tensor::SPtr outGrad)
{
    assert(out == mOutputs[0]);

    Layer::SPtr layer = createLayer<Nhwc2NchwLayer>(outGrad);

    return {{getInputs()[0], layer->getOutputs()[0]}};
}

void Nchw2NhwcLayer::execute(const std::vector<float*>& inputs,
                             const std::vector<float*>& outputs,
                             const InputDict& /*inputDict*/)
{
    float* in = inputs[0];
    float* out = outputs[0];
    std::vector<int> outShape = getOutputs()[0]->getShape();

    if (getInputs()[0]->getType() == MemoryType::kHOST_MEMORY)
        runNchw2NhwcHost(in, out, outShape.data());
#ifdef CUDA_AVAILABLE
    else
        cuda::runNchw2NhwcDevice(in, out, outShape.data());
#endif
}

}  // namespace layers

Tensor::SPtr nhwc2nchw(const Tensor::SPtr& tensor)
{
    if (tensor->getShape().size() != 4)
        throw std::runtime_error(
            "nhwc2nchw: input tensor must be 4-dimensional");

    Layer::SPtr layer = createLayer<layers::Nhwc2NchwLayer>(tensor);
    return layer->getOutputs()[0];
}

Tensor::SPtr nchw2nhwc(const Tensor::SPtr& tensor)
{
    if (tensor->getShape().size() != 4)
        throw std::runtime_error(
            "nchw2nhwc: input tensor must be 4-dimensional");

    Layer::SPtr layer = createLayer<layers::Nchw2NhwcLayer>(tensor);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr nhwc2nchw(const ITensorPtr& tensor)
{
    core::AbstractTensor::Ptr t = core::castITensorPtr(tensor);
    return makeAbstractTensor(core::nhwc2nchw(t->get()));
}

ITensorPtr nchw2nhwc(const ITensorPtr& tensor)
{
    core::AbstractTensor::Ptr t = core::castITensorPtr(tensor);
    return makeAbstractTensor(core::nchw2nhwc(t->get()));
}

}  // namespace graphdl
