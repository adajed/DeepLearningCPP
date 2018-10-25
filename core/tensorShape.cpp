#include "tensorShape.h"

namespace dll
{
namespace core
{
TensorShape::TensorShape(const Shape& shape) : mDims(shape) {}

TensorShape::TensorShape(const TensorShape& other) : mDims(other.mDims) {}

std::size_t TensorShape::count() const
{
    std::size_t count = 1;
    for (std::size_t dim : mDims) count *= dim;
    return count;
}

TensorShape::operator Shape() const { return mDims; }

TensorShape::iterator TensorShape::begin() { return mDims.begin(); }

TensorShape::iterator TensorShape::end() { return mDims.end(); }

}  // namespace core
}  // namespace dll
