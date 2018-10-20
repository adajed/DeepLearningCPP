#include "oper.h"

namespace dll
{
namespace core
{

std::vector<Tensor*> Oper::getInputs()
{
    return mInputs;
}

std::vector<Tensor*> Oper::getOutputs()
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
    for (Tensor* output : mOutputs)
        output->reset();
}

} // namespace core
} // namespace dll
