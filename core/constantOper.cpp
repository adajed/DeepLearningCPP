#include "constantOper.h"
#include "dll_ops.h"
#include "graph.h"

namespace dll
{
namespace core
{

Tensor::SPtr constant(float value, const Shape& shape)
{
    Oper::SPtr oper = std::make_shared<ConstantOper>(value, shape);
    getDefaultGraph()->insertOperation(oper);
    return oper->getOutputs()[0];
}

}  // namespace core

ITensorSPtr constant(float value, const Shape& shape)
{
    return ITensorSPtr(core::constant(value, shape));
}

}  // namespace dll
