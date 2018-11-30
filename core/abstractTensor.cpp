#include "abstractTensor.h"

#include <cassert>
#include <utility>

namespace graphdl
{
namespace core
{
AbstractTensor::AbstractTensor(Tensor::SPtr tensor) : mTensor(std::move(tensor))
{
}

std::string AbstractTensor::getName() const
{
    return mTensor->getName();
}

void AbstractTensor::setName(const std::string& name)
{
    mTensor->setName(name);
}

Shape AbstractTensor::getShape() const
{
    return mTensor->getShape();
}

HostTensor AbstractTensor::eval(const InputDict& inputs)
{
    std::set<Tensor::SPtr> tensors = mTensor->getNecessaryInputs();
    for (const auto& t : tensors)
    {
        if (inputs.count(t->getName()) == 0)
            throw std::runtime_error("eval: input \"" + t->getName() +
                                     "\" not provided");
        if (inputs.at(t->getName()).size() != t->getShape().getCount())
            throw std::runtime_error("eval: input \"" + t->getName() +
                                     "\" has wrong shape");
    }

    mTensor->eval(inputs);

    HostTensor output(mTensor->getShape().getCount());
    mTensor->getMemory().fill(output.data());
    return output;
}

Tensor::SPtr AbstractTensor::get() const
{
    return mTensor;
}

AbstractTensor::Ptr makeAbstractTensor(Tensor::SPtr tensor)
{
    static std::map<Tensor::SPtr, AbstractTensor::Ptr> sMap;

    if (sMap.count(tensor) == 0)
        sMap[tensor] = std::make_shared<AbstractTensor>(tensor);

    return sMap[tensor];
}

AbstractTensor::Ptr castITensorPtr(const ITensorPtr& itensor)
{
    return std::static_pointer_cast<AbstractTensor>(itensor);
}

}  // namespace core
}  // namespace graphdl
