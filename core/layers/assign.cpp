#include "assign.h"
#include <assert.h>
#include "dll_errors.h"
#include "dll_ops.h"
#include "graph.h"

namespace dll
{
namespace core
{
namespace layers
{
AssignOper::AssignOper(Tensor::SPtr dest, Tensor::SPtr src)
    : Oper({src}, {createTensor("", {0})}), mDest(dest)
{
    assert(dest->shape() == src->shape());
}

void AssignOper::executeOper(const InputDict& inputs)
{
    Tensor::SPtr src = mInputs[0].lock();
    Tensor::SPtr dest = mDest.lock();
    src->exec(inputs);

    Memory in = src->getMemory();
    Memory out = dest->getMemory();
    for (std::size_t pos = 0; pos < in.count(); ++pos) out[pos] = in[pos];
}

}  // namespace layers

Tensor::SPtr assign(Tensor::SPtr dest, Tensor::SPtr src)
{
    if (dest->shape() != src->shape()) throw errors::NotMatchingShapesError();

    Oper::SPtr oper = std::make_shared<layers::AssignOper>(dest, src);
    getDefaultGraph()->insertOperation(oper);
    return oper->getOutputs()[0];
}

}  // namespace core

ITensorSPtr assign(ITensorSPtr dest, ITensorSPtr src)
{
    core::Tensor::SPtr destT = std::static_pointer_cast<core::Tensor>(dest);
    core::Tensor::SPtr srcT = std::static_pointer_cast<core::Tensor>(src);
    return ITensorSPtr(core::assign(destT, srcT));
}

}  // namespace dll
