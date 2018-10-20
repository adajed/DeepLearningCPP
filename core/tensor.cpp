#include "oper.h"

namespace dll
{
namespace core
{

std::string Tensor::getName() const
{
    return mName;
}

void Tensor::setName(const std::string& name)
{
    mName = name;
}

Shape Tensor::getShape() const
{
    return mShape;
}

void Tensor::setShape(const Shape& shape)
{
    mShape = shape;
}

HostTensor Tensor::eval(InputDict const& inputs)
{
    exec(inputs);
    return HostTensor{};
}

void Tensor::setHostTensor(HostTensor tensor)
{
}

void Tensor::exec(const InputDict& inputs)
{
    if (!mIsEvaluated)
    {
        mOper->exec(inputs);
        mIsEvaluated = true;
    }
}

void Tensor::reset()
{
    mIsEvaluated = false;
    for (Oper* op : mOutputOps)
        op->reset();
}

Tensor::~Tensor()
{
    for (Oper* op : mOutputOps)
        delete op;
}

} // namespace core
} // namespace dll
