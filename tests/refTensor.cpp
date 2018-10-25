#include "refTensor.h"

#include <assert.h>

RefTensor::RefTensor() : mValues(0), mCount(0), mShape({}) {}

RefTensor::RefTensor(const TensorShape& shape)
    : mValues(shape.count()), mCount(shape.count()), mShape(shape)
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

float& RefTensor::operator[](const std::vector<unsigned int>& point)
{
    assert(mShape.size() == point.size());

    std::size_t pos = 0;
    for (std::size_t i = 0; i < point.size(); ++i)
    {
        assert(mShape[i] > point[i]);
        pos *= mShape[i];
        pos += point[i];
    }
    return at(pos);
}

const float& RefTensor::operator[](const std::vector<unsigned int>& point) const
{
    assert(mShape.size() == point.size());

    std::size_t pos = 0;
    for (std::size_t i = 0; i < point.size(); ++i)
    {
        assert(mShape[i] > point[i]);
        pos *= mShape[i];
        pos += point[i];
    }
    return at(pos);
}

std::size_t RefTensor::count() const { return mCount; }

TensorShape RefTensor::shape() const { return mShape; }

void RefTensor::fillRandomly(RandGen& gen)
{
    for (std::size_t i = 0; i < mCount; ++i)
    {
        at(i) = gen();
    }
}

HostTensor RefTensor::toHostTensor()
{
    HostTensor t{nullptr, mCount};
    t.values = new float[mCount];

    for (std::size_t i = 0; i < mCount; ++i) t.values[i] = mValues[i];
    return t;
}

std::ostream& operator<<(std::ostream& stream, const RefTensor& tensor)
{
    for (std::size_t i = 0; i < tensor.count(); ++i)
        stream << tensor.at(i) << " ";
    return stream;
}
