#include "queue.h"

#include "abstractTensor.h"
#include "graph.h"

namespace graphdl
{
namespace core
{
namespace layers
{
QueueLayer::QueueLayer(ID id, const std::vector<Tensor::SPtr>& ops)
    : Layer(id, ops, {createTensor("", {0}, MemoryType::kHOST_MEMORY)})

{
    for (auto& op : ops) mOps.push_back(Tensor::SPtr(op));
}

void QueueLayer::eval(const InputDict& inputDict)
{
    if (!mIsEvaluated)
    {
        for (auto& op : mOps) op.lock()->eval(inputDict);
        mIsEvaluated = true;
    }
}

void QueueLayer::execute(const std::vector<float*>& /*inputs*/,
                         const std::vector<float*>& /*outputs*/,
                         const InputDict& /*inputDict*/)
{
}

}  // namespace layers

Tensor::SPtr queue(const std::vector<Tensor::SPtr>& ops)
{
    for (const Tensor::SPtr& t : ops)
        if (t->getShape().getCount() != 0)
            throw std::runtime_error("Wrong shape");

    Layer::SPtr layer = createLayer<layers::QueueLayer>(ops);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr queue(const std::vector<ITensorPtr>& tensors)
{
    std::vector<core::Tensor::SPtr> ts;
    ts.reserve(tensors.size());
    for (const ITensorPtr& iTensor : tensors)
        ts.push_back(core::castITensorPtr(iTensor)->get());

    return makeAbstractTensor(core::queue(ts));
}

}  // namespace graphdl
