#include "refTensor.h"

#include <assert.h>

Coord::Coord(const std::vector<unsigned>& values) : mValues(values) {}
Coord::Coord(std::initializer_list<unsigned> list) : mValues(list) {}

unsigned Coord::size() const
{
    return mValues.size();
}

unsigned& Coord::operator[](size_t pos)
{
    return mValues[pos];
}
const unsigned& Coord::operator[](size_t pos) const
{
    return mValues[pos];
}

Coord_iterator::Coord_iterator(Coord c, Coord shape) : mCoord(c), mShape(shape)
{
}

Coord_iterator Coord_iterator::operator++()
{
    Coord_iterator it = *this;

    unsigned p = mCoord.size() - 1;
    mCoord[p]++;
    while (p > 0)
    {
        if (mCoord[p] == mShape[p])
        {
            mCoord[p--] = 0;
            ++mCoord[p];
        }
        else
            break;
    }

    return it;
}

Coord_iterator Coord_iterator::operator++(int junk)
{
    unsigned p = mCoord.size() - 1;
    mCoord[p]++;
    while (p > 0)
    {
        if (mCoord[p] == mShape[p])
        {
            mCoord[p--] = 0;
            ++mCoord[p];
        }
        else
            break;
    }

    return *this;
}

bool Coord_iterator::operator==(const Coord_iterator& it) const
{
    if (mCoord.size() != it.mCoord.size()) return false;

    for (unsigned i = 0; i < mCoord.size(); ++i)
        if (mCoord[i] != it.mCoord[i]) return false;

    return true;
}

bool Coord_iterator::operator!=(const Coord_iterator& it) const
{
    return !operator==(it);
}

Coord& Coord_iterator::operator()()
{
    return mCoord;
}

RefTensor::RefTensor() : mValues(0), mCount(0), mShape({}) {}

RefTensor::RefTensor(const TensorShape& shape)
    : mValues(shape.getCount()), mCount(shape.getCount()), mShape(shape)
{
}

float& RefTensor::at(std::size_t pos)
{
    assert(pos < mValues.size());
    return mValues[pos];
}

const float& RefTensor::at(std::size_t pos) const
{
    assert(pos < mValues.size());
    return mValues[pos];
}

float& RefTensor::operator[](const Coord& c)
{
    assert(mShape.size() == c.size());

    std::size_t pos = 0;
    for (std::size_t i = 0; i < c.size(); ++i)
    {
        assert(mShape[i] > c[i]);
        pos *= mShape[i];
        pos += c[i];
    }
    return at(pos);
}

const float& RefTensor::operator[](const Coord& c) const
{
    assert(mShape.size() == c.size());

    std::size_t pos = 0;
    for (std::size_t i = 0; i < c.size(); ++i)
    {
        assert(mShape[i] > c[i]);
        pos *= mShape[i];
        pos += c[i];
    }
    return at(pos);
}

Coord_iterator RefTensor::begin()
{
    std::vector<unsigned> c;
    std::vector<unsigned> shape;
    for (unsigned i = 0; i < mShape.size(); ++i)
    {
        c.push_back(0);
        shape.push_back(mShape[i]);
    }
    return Coord_iterator(Coord(c), Coord(shape));
}

Coord_iterator RefTensor::end()
{
    std::vector<unsigned> c;
    std::vector<unsigned> shape;
    for (unsigned i = 0; i < mShape.size(); ++i)
    {
        c.push_back(0);
        shape.push_back(mShape[i]);
    }
    c[0] = mShape[0];
    return Coord_iterator(Coord(c), Coord(shape));
}

std::size_t RefTensor::getCount() const
{
    return mCount;
}

TensorShape RefTensor::shape() const
{
    return mShape;
}

void RefTensor::fillRandomly(RandGen& gen)
{
    for (std::size_t i = 0; i < mCount; ++i)
    {
        at(i) = gen();
    }
}

HostTensor RefTensor::toHostTensor()
{
    HostTensor t(mCount);

    for (std::size_t i = 0; i < mCount; ++i) t[i] = mValues[i];
    return t;
}

std::ostream& operator<<(std::ostream& stream, const RefTensor& tensor)
{
    for (std::size_t i = 0; i < tensor.getCount(); ++i)
        stream << tensor.at(i) << " ";
    return stream;
}
