#include "constant.h"

#include "abstractTensor.h"
#include "graph.h"
#include "graphdl_ops.h"

namespace graphdl
{
namespace core
{
namespace layers
{
ConstantLayer::ConstantLayer(ID id, float value, const TensorShape& shape)
    : Layer(id, {}, {createTensor("", shape)}), mValue(value)
{
}

void ConstantLayer::initialize()
{
    Memory out = mOutputs[0]->getMemory();
    for (std::size_t i = 0; i < out.getCount(); ++i) out[i] = mValue;
}

//! This method does nothing, because tensor is already
//!     filled during initialize.
void ConstantLayer::execute(const InputDict& /*inputs*/) {}

}  // namespace layers

Tensor::SPtr constant(float value, const TensorShape& shape)
{
    Layer::SPtr layer = createLayer<layers::ConstantLayer>(value, shape);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr constant(float value, const Shape& shape)
{
    return makeAbstractTensor(core::constant(value, shape));
}

}  // namespace graphdl
