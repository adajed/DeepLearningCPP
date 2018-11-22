#include "constant.h"

#include "abstractTensor.h"
#include "graph.h"
#include "graphdl_ops.h"
#include "memory.h"

namespace graphdl
{
namespace core
{
namespace layers
{
ConstantLayer::ConstantLayer(ID id, float value, const TensorShape& shape,
                             MemoryType type)
    : Layer(id, {}, {createTensor("", shape, type)}), mValue(value)
{
}

void ConstantLayer::initialize()
{
    float* out = mOutputs[0]->getMemory().getValues();
    std::size_t size = mOutputs[0]->getMemory().getCount();

    if (mOutputs[0]->getType() == MemoryType::kHOST_MEMORY)
        for (std::size_t i = 0; i < size; ++i) out[i] = mValue;
#ifdef CUDA_AVAILABLE
    else
        cuda::fillWithValue(size, out, mValue);
#endif
}

//! This method does nothing, because tensor is already
//!     filled during initialize.
void ConstantLayer::execute(const InputDict& /*inputs*/) {}

}  // namespace layers

Tensor::SPtr constant(float value, const TensorShape& shape, MemoryType type)
{
    Layer::SPtr layer = createLayer<layers::ConstantLayer>(value, shape, type);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr constant(float value, const Shape& shape, MemoryLocation location)
{
    core::MemoryType type = core::memoryLocationToType(location);
    return makeAbstractTensor(core::constant(value, shape, type));
}

ITensorPtr scalar(float value, MemoryLocation location)
{
    return constant(value, {}, location);
}

}  // namespace graphdl
