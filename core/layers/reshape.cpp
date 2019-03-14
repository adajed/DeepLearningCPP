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

void ReshapeLayer::execute(const std::vector<float*>& inputs,
                           const std::vector<float*>& outputs,
                           const InputDict& /*inputDict*/)
{
    Tensor::SPtr tX = getInputs()[0];
    float* x = inputs[0];
    float* y = outputs[0];
    if (tX->getType() == MemoryType::kHOST_MEMORY)
        std::memcpy(y, x, tX->getCount() * sizeof(float));
#ifdef CUDA_AVAILABLE
    else  // input->getType() == MemoryType::kDEVICE_MEMORY
        cuda::utils::copy(y, x, tX->getCount());
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
