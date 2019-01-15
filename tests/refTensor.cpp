#include "refTensor.h"

#include <assert.h>

Coord::Coord(const std::vector<int>& values) : mValues(values) {}
Coord::Coord(std::initializer_list<int> list) : mValues(list) {}

Coord Coord::operator+(const Coord& c) const
{
    assert(mValues.size() == c.mValues.size());

    std::vector<int> v = mValues;
    for (unsigned i = 0; i < mValues.size(); ++i) v[i] += c.mValues[i];

    return Coord(v);
}

unsigned Coord::size() const
{
    return mValues.size();
}

int& Coord::operator[](size_t pos)
{
    return mValues[pos];
}
const int& Coord::operator[](size_t pos) const
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

bool isInside(const Coord& c, const TensorShape& shape)
{
    assert(c.size() == shape.size());

    for (unsigned i = 0; i < c.size(); ++i)
        if (c[i] < 0 || c[i] >= int(shape[i])) return false;
    return true;
}

RefTensor::RefTensor() : mValues(0), mCount(0), mShape({}) {}

RefTensor::RefTensor(const TensorShape& shape)
    : mValues(shape.getCount()), mCount(shape.getCount()), mShape(shape)
{
}

RefTensor::RefTensor(const TensorShape& shape, RandGen& gen) : RefTensor(shape)
{
    fillRandomly(gen);
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
        assert(int(mShape[i]) > c[i]);
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
        assert(int(mShape[i]) > c[i]);
        pos *= mShape[i];
        pos += c[i];
    }
    return at(pos);
}

Coord_iterator RefTensor::begin()
{
    return shapeBegin(mShape);
}

Coord_iterator RefTensor::end()
{
    return shapeEnd(mShape);
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

RefTensor RefTensor::slice(Coord start, const TensorShape& shape) const
{
    RefTensor tensor(shape);

    for (Coord_iterator it = tensor.begin(); it != tensor.end(); ++it)
    {
        if (isInside(start + it(), mShape))
            tensor[it()] = this->operator[](start + it());
        else
            tensor[it()] = 0.;
    }

    return tensor;
}

std::ostream& operator<<(std::ostream& stream, const RefTensor& tensor)
{
    for (std::size_t i = 0; i < tensor.getCount(); ++i)
        stream << tensor.at(i) << " ";
    return stream;
}

Coord_iterator shapeBegin(const TensorShape& shape)
{
    std::vector<int> c, s;
    for (unsigned i = 0; i < shape.size(); ++i)
    {
        c.push_back(0);
        s.push_back(shape[i]);
    }
    return Coord_iterator(Coord(c), Coord(s));
}

Coord_iterator shapeEnd(const TensorShape& shape)
{
    std::vector<int> c, s;
    for (unsigned i = 0; i < shape.size(); ++i)
    {
        c.push_back(0);
        s.push_back(shape[i]);
    }
    c[0] = shape[0];
    return Coord_iterator(Coord(c), Coord(s));
}
