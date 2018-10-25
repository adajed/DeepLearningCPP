#include "addOper.h"
#include "divOper.h"
#include "dll_ops.h"
#include "graph.h"
#include "mulOper.h"
#include "subOper.h"

namespace dll
{
ITensorSPtr add(ITensorSPtr t1, ITensorSPtr t2)
{
    core::Tensor::SPtr tensor1 = std::static_pointer_cast<core::Tensor>(t1);
    core::Tensor::SPtr tensor2 = std::static_pointer_cast<core::Tensor>(t2);
    std::shared_ptr<core::AddOper> oper =
        std::make_shared<core::AddOper>(tensor1, tensor2);
    core::getDefaultGraph()->insertOperation(core::Oper::SPtr(oper));
    return ITensorSPtr(oper->getOutputs()[0]);
}

ITensorSPtr operator+(ITensorSPtr t1, ITensorSPtr t2) { return add(t1, t2); }

ITensorSPtr sub(ITensorSPtr t1, ITensorSPtr t2)
{
    core::Tensor::SPtr tensor1 = std::static_pointer_cast<core::Tensor>(t1);
    core::Tensor::SPtr tensor2 = std::static_pointer_cast<core::Tensor>(t2);
    std::shared_ptr<core::SubOper> oper =
        std::make_shared<core::SubOper>(tensor1, tensor2);
    core::getDefaultGraph()->insertOperation(core::Oper::SPtr(oper));
    return ITensorSPtr(oper->getOutputs()[0]);
}

ITensorSPtr operator-(ITensorSPtr t1, ITensorSPtr t2) { return sub(t1, t2); }

ITensorSPtr mul(ITensorSPtr t1, ITensorSPtr t2)
{
    core::Tensor::SPtr tensor1 = std::static_pointer_cast<core::Tensor>(t1);
    core::Tensor::SPtr tensor2 = std::static_pointer_cast<core::Tensor>(t2);
    std::shared_ptr<core::MulOper> oper =
        std::make_shared<core::MulOper>(tensor1, tensor2);
    core::getDefaultGraph()->insertOperation(core::Oper::SPtr(oper));
    return ITensorSPtr(oper->getOutputs()[0]);
}

ITensorSPtr operator*(ITensorSPtr t1, ITensorSPtr t2) { return mul(t1, t2); }

ITensorSPtr div(ITensorSPtr t1, ITensorSPtr t2)
{
    core::Tensor::SPtr tensor1 = std::static_pointer_cast<core::Tensor>(t1);
    core::Tensor::SPtr tensor2 = std::static_pointer_cast<core::Tensor>(t2);
    std::shared_ptr<core::DivOper> oper =
        std::make_shared<core::DivOper>(tensor1, tensor2);
    core::getDefaultGraph()->insertOperation(core::Oper::SPtr(oper));
    return ITensorSPtr(oper->getOutputs()[0]);
}

ITensorSPtr operator/(ITensorSPtr t1, ITensorSPtr t2) { return div(t1, t2); }

}  // namespace dll
