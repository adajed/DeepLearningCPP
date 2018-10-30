#include "abstractTensor.h"
#include <assert.h>

namespace graphdl
{
namespace core
{
AbstractTensor::AbstractTensor(Tensor::SPtr tensor) : mTensor(tensor) {}

std::string AbstractTensor::getName() const { return mTensor->getName(); }

void AbstractTensor::setName(const std::string& name)
{
    mTensor->setName(name);
}

Shape AbstractTensor::getShape() const { return mTensor->getShape(); }

void AbstractTensor::eval(const InputDict& inputs, HostTensor hostTensor)
{
    assert(mTensor->getMemory().getCount() >= hostTensor.size());
    mTensor->eval(inputs);
    mTensor->getMemory().fill(hostTensor.data());
}

AbstractTensor::Ptr makeAbstract(Tensor::SPtr tensor)
{
    return std::make_unique<AbstractTensor>(tensor);
}

}  // namespace core
}  // namespace graphdl
