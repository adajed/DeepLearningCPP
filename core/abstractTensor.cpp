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

Tensor::SPtr AbstractTensor::get() const { return mTensor; }

AbstractTensor::Ptr makeAbstractTensor(Tensor::SPtr tensor)
{
    static std::map<Tensor::SPtr, AbstractTensor::Ptr> sMap;

    if (sMap.count(tensor) == 0)
        sMap[tensor] = std::make_shared<AbstractTensor>(tensor);

    return sMap[tensor];
}

AbstractTensor::Ptr castITensorPtr(ITensorPtr itensor)
{
    return std::static_pointer_cast<AbstractTensor>(itensor);
}

}  // namespace core
}  // namespace graphdl
