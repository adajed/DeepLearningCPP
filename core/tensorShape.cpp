#include "tensorShape.h"

namespace dll
{
namespace core
{

TensorShape::TensorShape(const Shape& shape)
    : mDims(shape)
{}

TensorShape::TensorShape(const TensorShape& other)
    : mDims(other.mDims)
{}

TensorShape::TensorShape(std::initializer_list<unsigned> list)
    : mDims(list)
{
}

bool TensorShape::operator ==(const TensorShape& other) const
{
    if (mDims.size() != other.mDims.size())
        return false;
    for (unsigned i = 0; i < mDims.size(); ++i)
        if (mDims[i] != other.mDims[i])
            return false;
    return true;
}

unsigned& TensorShape::operator [](std::size_t pos)
{
    return mDims[pos];
}

const unsigned& TensorShape::operator [](std::size_t pos) const
{
    return mDims[pos];
}

unsigned TensorShape::size() const
{
    return mDims.size();
}

std::size_t TensorShape::count() const
{
    std::size_t count = 1;
    for (std::size_t dim : mDims)
        count *= dim;
    return count;
}

TensorShape::operator Shape() const
{
    return mDims;
}

TensorShape::iterator TensorShape::begin()
{
    return mDims.begin();
}

TensorShape::iterator TensorShape::end()
{
    return mDims.end();
}

} // namespace core
} // namespace dll
