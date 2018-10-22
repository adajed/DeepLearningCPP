#include <assert.h>
#include <iostream>

#include "memory.h"

namespace dll
{
namespace core
{

Memory::Memory(MemoryType type, size_t count)
    : mType(type), mValues(nullptr), mCount(count)
{}

bool Memory::isAllocated() const
{
    return mValues != nullptr;
}

float* Memory::getValues()
{
    return mValues;
}

const float* Memory::getValues() const
{
    return mValues;
}

size_t Memory::getCount() const
{
    return mCount;
}

bool Memory::allocate()
{
    assert(!isAllocated());
    if (mType == MemoryType::kHOST_MEMORY)
    {
        mValues = new float[mCount];
        return true;
    }
    else // mType == MemoryType::kDEVICE_MEMORY
    {
        // TODO: allocate memory on device
        return false;
    }

    // you shouldn't be here
    return false;
}

void Memory::free()
{
    if (!isAllocated())
        return;

    if (mType == MemoryType::kHOST_MEMORY)
    {
        delete [] mValues;
    }
    else // mType == MemoryType::kDEVICE_MEMORY
    {
        // TODO: free memory on device
    }
}

void Memory::fill(HostTensor hostTensor) const
{
    assert(isAllocated());
    assert(getCount() == hostTensor.count);

    for (std::size_t i = 0; i < hostTensor.count; ++i)
        hostTensor.values[i] = mValues[i];
}

Memory::~Memory()
{
    free();
}

} // namespace core
} // namespace dll
