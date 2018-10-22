#include "oper.h"

namespace dll
{
namespace core
{

Oper::ID Oper::getID() const
{
    return mID;
}

std::vector<Tensor::SPtr> Oper::getInputs()
{
    std::vector<Tensor::SPtr> inputs;
    for (Tensor::WeakPtr input : inputs)
        inputs.push_back(input.lock());
    return inputs;
}

std::vector<Tensor::SPtr> Oper::getOutputs()
{
    return mOutputs;
}

void Oper::exec(const InputDict& inputs)
{
    if (!mIsEvaluated)
    {
        // calculate actual operation
        executeOper(inputs);
        mIsEvaluated = true;
    }
}

void Oper::reset()
{
    mIsEvaluated = false;
    for (Tensor::SPtr output : mOutputs)
        output->reset();
}

} // namespace core
} // namespace dll
