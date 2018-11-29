#include "reshape.h"

#include "abstractTensor.h"
#ifdef CUDA_AVAILABLE
#include "cuda/utils.h"
#endif
#include "graph.h"
#include "layer.h"

#include <cstring>

namespace graphdl
{
namespace core
{
namespace layers
{
ReshapeLayer::ReshapeLayer(ID id, const Tensor::SPtr& t,
                           const TensorShape& shape)
    : DifferentiableLayer(id, {t}, {createTensor("", shape, t->getType())})
{
}

Layer::TensorMap ReshapeLayer::gradients(Tensor::SPtr /* out */,
                                         Tensor::SPtr outGrad)
{
    Tensor::SPtr input = mInputs[0].lock();
    Layer::SPtr layer = createLayer<ReshapeLayer>(outGrad, input->getShape());
    return {{input, layer->getOutputs()[0]}};
}

void ReshapeLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr input = mInputs[0].lock();
    input->eval(inputs);

    float* in = input->getMemory().getValues();
    float* out = mOutputs[0]->getMemory().getValues();
    if (input->getType() == MemoryType::kHOST_MEMORY)
        std::memcpy(out, in, input->getShape().getCount() * sizeof(float));
#ifdef CUDA_AVAILABLE
    else  // input->getType() == MemoryType::kDEVICE_MEMORY
        cuda::utils::copy(out, in, input->getShape().getCount());
#endif
}

}  // namespace layers

Tensor::SPtr reshape(const Tensor::SPtr& t, const TensorShape& shape)
{
    if (t->getShape().getCount() != shape.getCount())
        throw std::runtime_error("reshape: Shapes don\'t match");

    Layer::SPtr layer = createLayer<layers::ReshapeLayer>(t, shape);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr reshape(const ITensorPtr& iTensor, const Shape& shape)
{
    core::AbstractTensor::Ptr t = core::castITensorPtr(iTensor);
    return makeAbstractTensor(core::reshape(t->get(), shape));
}

}  // namespace graphdl
