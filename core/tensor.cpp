#include "oper.h"

namespace dll
{
namespace core
{

Tensor::ID Tensor::getID() const
{
    return mID;
}

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

TensorShape Tensor::getTensorShape() const
{
    return mShape;
}

void Tensor::setTensorShape(const TensorShape& shape)
{
    mShape = shape;
}

Memory Tensor::getMemory()
{
    return mMemory;
}

void Tensor::eval(InputDict const& inputs, HostTensor* hostTensor)
{
    exec(inputs);
    mMemory.fill(hostTensor);
}

void Tensor::exec(const InputDict& inputs)
{
    if (!mIsEvaluated)
    {
        mOper->exec(inputs);
        mIsEvaluated = true;
    }
}

bool Tensor::allocateMemory()
{
    return mMemory.allocate();
}

void Tensor::freeMemory()
{
    mMemory.free();
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

    mMemory.free();
}

} // namespace core
} // namespace dll
