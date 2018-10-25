#include "oper.h"

namespace dll
{
namespace core
{
Tensor::ID Tensor::getID() const { return mID; }

std::string Tensor::getName() const { return mName; }

void Tensor::setName(const std::string& name) { mName = name; }

Shape Tensor::getShape() const { return mShape; }

void Tensor::setShape(const Shape& shape) { mShape = shape; }

TensorShape Tensor::shape() const { return mShape; }

void Tensor::setTensorShape(const TensorShape& shape) { mShape = shape; }

Oper::SPtr Tensor::getOper() const { return mOper.lock(); }

void Tensor::setOper(Oper::SPtr oper) { mOper = Oper::WeakPtr(oper); }

Memory Tensor::getMemory() { return mMemory; }

void Tensor::eval(InputDict const& inputs, HostTensor hostTensor)
{
    exec(inputs);
    mMemory.fill(hostTensor);
}

void Tensor::exec(const InputDict& inputs)
{
    if (!mIsEvaluated)
    {
        mOper.lock()->exec(inputs);
        mIsEvaluated = true;
    }
}

bool Tensor::allocateMemory() { return mMemory.allocate(); }

void Tensor::freeMemory() { mMemory.free(); }

void Tensor::reset()
{
    mIsEvaluated = false;
    for (Oper::SPtr op : mOutputOps) op->reset();
}

Tensor::~Tensor() { mMemory.free(); }

}  // namespace core
}  // namespace dll
