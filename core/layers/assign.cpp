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

void AssignLayer::execute(const std::vector<float*>& inputs,
                          const std::vector<float*>& /*outputs*/,
                          const InputDict& /*inputDict*/)
{
    Tensor::SPtr tDest = mDest.lock();

    size_t size = tDest->getCount();
    float* src = inputs[0];
    float* dest = tDest->getMemory().getValues();

    if (tDest->getType() == MemoryType::kHOST_MEMORY)
        std::memcpy(dest, src, size * sizeof(float));
#ifdef CUDA_AVAILABLE
    else
        cuda::assignDevice(dest, src, size);
#endif
}

}  // namespace layers

Tensor::SPtr assign(const Tensor::SPtr& dest, const Tensor::SPtr& src)
{
    // check is dest is weights tensor
    const WeightsNamespaces& namespaces =
        getDefaultGraph()->getWeightsNamespaces();
    if (!namespaces.contains(dest->getLayer()))
        throw std::runtime_error("assign: destination must be weights");

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
