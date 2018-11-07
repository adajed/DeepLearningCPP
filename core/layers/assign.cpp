#include "assign.h"

#include "abstractTensor.h"
#include "graph.h"
#include "graphdl_ops.h"

#include <cassert>

namespace graphdl
{
namespace core
{
namespace layers
{
AssignLayer::AssignLayer(ID id, const Tensor::SPtr& dest,
                         const Tensor::SPtr& src)
    : Layer(id, {src}, {createTensor("", {0}, MemoryType::kHOST_MEMORY)}),
      mDest(dest)
{
    assert(dest->getShape() == src->getShape());
}

void AssignLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr src = mInputs[0].lock();
    Tensor::SPtr dest = mDest.lock();
    src->eval(inputs);

    float* in = src->getMemory().getValues();
    float* out = dest->getMemory().getValues();
    std::size_t size = src->getMemory().getCount();

    if (src->getType() == MemoryType::kHOST_MEMORY)
    {
        for (std::size_t pos = 0; pos < size; ++pos) out[pos] = in[pos];
    }
#ifdef CUDA_AVAILABLE
    else
        cuda::assignDevice(out, in, size);
#endif
}

}  // namespace layers

Tensor::SPtr assign(const Tensor::SPtr& dest, const Tensor::SPtr& src)
{
    if (dest->getShape() != src->getShape())
        throw std::runtime_error("Shapes don\'t match");
    if (dest->getType() != src->getType())
        throw std::runtime_error("Tensor must be on the same device type");

    Layer::SPtr layer = createLayer<layers::AssignLayer>(dest, src);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr assign(const ITensorPtr& dest, const ITensorPtr& src)
{
    core::AbstractTensor::Ptr aDest = core::castITensorPtr(dest);
    core::AbstractTensor::Ptr aSrc = core::castITensorPtr(src);
    return makeAbstractTensor(core::assign(aDest->get(), aSrc->get()));
}

}  // namespace graphdl
