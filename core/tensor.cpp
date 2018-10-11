#include "tensor.h"

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
    return HostTensor{};
}

} // namespace core
} // namespace dll
