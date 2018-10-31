#include <assert.h>
#include <cstring>
#include <iostream>

#include "memory.h"

namespace graphdl
{
namespace core
{
Memory::Memory(MemoryType type, size_t count)
    : mType(type), mValues(nullptr), mCount(count)
{
}

float* Memory::getValues() { return mValues; }

const float* Memory::getValues() const { return mValues; }

float& Memory::operator[](std::size_t pos) { return mValues[pos]; }

const float& Memory::operator[](std::size_t pos) const { return mValues[pos]; }

size_t Memory::getCount() const { return mCount; }

void Memory::fill(float* memory) const
{
    assert(isAllocated());
    std::memcpy(memory, mValues, sizeof(float) * getCount());
}

bool Memory::isAllocated() const { return mValues != nullptr; }

bool Memory::allocate()
{
    assert(!isAllocated());
    if (mType == MemoryType::kHOST_MEMORY)
    {
        mValues = new float[mCount];
        return true;
    }
    else  // mType == MemoryType::kDEVICE_MEMORY
    {
        // TODO: allocate memory on device
        throw std::runtime_error("GPU support not implemented, please use CPU");
    }

    // you shouldn't be here
    return false;
}

void Memory::free()
{
    if (!isAllocated()) return;

    if (mType == MemoryType::kHOST_MEMORY)
    {
        delete[] mValues;
    }
    else  // mType == MemoryType::kDEVICE_MEMORY
    {
        // TODO: free memory on device
        throw std::runtime_error("GPU support not implemented, please use CPU");
    }
}

}  // namespace core
}  // namespace graphdl
