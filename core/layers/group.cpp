#include "group.h"

#include "abstractTensor.h"
#include "graph.h"
#include "graphdl_ops.h"
#include "layer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
GroupLayer::GroupLayer(ID id, const std::vector<Tensor::SPtr>& tensors)
    : Layer(id, tensors, {createTensor("", {0}, MemoryType::kHOST_MEMORY)})
{
}

// already calculated in eval
void GroupLayer::execute(const std::vector<float*>& /*inputs*/,
                         const std::vector<float*>& /*outputs*/,
                         const InputDict& /*inputDict*/)
{
}

}  // namespace layers

Tensor::SPtr group(const std::vector<Tensor::SPtr>& tensors)
{
    for (const Tensor::SPtr& t : tensors)
        if (t->getShape().getCount() != 0)
            throw std::runtime_error("Wrong shape");

    Layer::SPtr layer = createLayer<layers::GroupLayer>(tensors);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr group(const std::vector<ITensorPtr>& tensors)
{
    std::vector<core::Tensor::SPtr> ts;
    ts.reserve(tensors.size());
    for (const ITensorPtr& iTensor : tensors)
        ts.push_back(core::castITensorPtr(iTensor)->get());

    return makeAbstractTensor(core::group(ts));
}

}  // namespace graphdl
