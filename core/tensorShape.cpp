#include "tensorShape.h"

#include <cassert>
#include <utility>

namespace graphdl
{
namespace core
{
TensorShape::TensorShape(Shape shape)
{
    mDims.reserve(mDims.size());
    for (int i : shape) mDims.push_back(int(i));
}

TensorShape::TensorShape(std::vector<int> vals) : mDims(std::move(vals))
{
    for (int i : mDims) assert(i >= 0);
}

TensorShape::TensorShape(std::initializer_list<int> list) : mDims(list)
{
    for (int i : mDims) assert(i >= 0);
}

bool TensorShape::operator==(const TensorShape& other) const
{
    if (mDims.size() != other.mDims.size()) return false;
    for (unsigned i = 0; i < mDims.size(); ++i)
        if (mDims[i] != other.mDims[i]) return false;
    return true;
}

bool TensorShape::operator!=(const TensorShape& other) const
{
    return !operator==(other);
}

int& TensorShape::operator[](std::size_t pos)
{
    return mDims[pos];
}

const int& TensorShape::operator[](std::size_t pos) const
{
    return mDims[pos];
}

unsigned TensorShape::size() const
{
    return mDims.size();
}

std::size_t TensorShape::getCount() const
{
    std::size_t count = 1;
    for (std::size_t dim : mDims) count *= dim;
    return count;
}

TensorShape::operator Shape() const
{
    Shape s(mDims.size(), 0);
    for (unsigned i = 0; i < mDims.size(); ++i) s[i] = unsigned(mDims[i]);
    return s;
}

TensorShape::operator std::vector<int>() const
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

TensorShape TensorShape::subshape(int start, int size)
{
    assert(start >= 0);
    assert(start + size <= mDims.size());

    std::vector<int> s(size);
    for (int i = 0; i < size; ++i) s[i] = mDims[start + i];
    return TensorShape(s);
}

}  // namespace core
}  // namespace graphdl
