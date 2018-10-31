#include "assign.h"
#include <assert.h>
#include "abstractTensor.h"
#include "graph.h"
#include "graphdl_ops.h"

namespace graphdl
{
namespace core
{
namespace layers
{
AssignLayer::AssignLayer(ID id, Tensor::SPtr dest, Tensor::SPtr src)
    : Layer(id, {src}, {createTensor("", {0})}), mDest(dest)
{
    assert(dest->getShape() == src->getShape());
}

void AssignLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr src = mInputs[0].lock();
    Tensor::SPtr dest = mDest.lock();
    src->eval(inputs);

    Memory in = src->getMemory();
    Memory out = dest->getMemory();
    for (std::size_t pos = 0; pos < in.getCount(); ++pos) out[pos] = in[pos];
}

}  // namespace layers

Tensor::SPtr assign(Tensor::SPtr dest, Tensor::SPtr src)
{
    if (dest->getShape() != src->getShape())
        throw std::runtime_error("Shapes don\'t match");

    Layer::SPtr layer = createLayer<layers::AssignLayer>(dest, src);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr assign(ITensorPtr dest, ITensorPtr src)
{
    core::AbstractTensor::Ptr aDest = core::castITensorPtr(dest);
    core::AbstractTensor::Ptr aSrc = core::castITensorPtr(src);
    return makeAbstractTensor(core::assign(aDest->get(), aSrc->get()));
}

}  // namespace graphdl
