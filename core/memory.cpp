#include "memory.h"
#include <assert.h>
#include <cstring>
#include <stdexcept>

#ifdef CUDA_SUPPORT
#include <cuda.h>
#endif

namespace graphdl
{
namespace core
{
Memory::Memory(MemoryType type, size_t count)
    : mType(type), mValues(nullptr), mCount(count)
{
}

MemoryType Memory::getType() const { return mType; }

float* Memory::getValues() { return mValues; }

const float* Memory::getValues() const { return mValues; }

size_t Memory::getCount() const { return mCount; }

void Memory::fill(float* memory) const
{
    assert(isAllocated());

    if (mType == MemoryType::kHOST_MEMORY)
    {
        std::memcpy(memory, mValues, sizeof(float) * getCount());
    }
    else  // mType == MemoryType::kHOST_MEMORY
    {
#ifdef CUDA_SUPPORT
        cudaMemcpy(memory, mValues, sizeof(float) * getCount(),
                   cudaMemcpyDeviceToHost);
#else
        throw std::runtime_error("GPU support not implemented, please use CPU");
#endif
    }
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
#ifdef CUDA_SUPPORT
        if (cudaMallocManaged((void**)&mValues, mCount * sizeof(float)) !=
            cudaSuccess)
            return false;
        return true;
#else
        throw std::runtime_error("GPU support not implemented, please use CPU");
#endif
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
#ifdef CUDA_SUPPORT
        cudaFree(mValues);
#else
        throw std::runtime_error("GPU support not implemented, please use CPU");
#endif
    }
}

MemoryType memoryLocationToType(MemoryLocation location)
{
    switch (location)
    {
        case MemoryLocation::kHOST:
            return MemoryType::kHOST_MEMORY;
        case MemoryLocation::kDEVICE:
            return MemoryType::kDEVICE_MEMORY;
    }

    // you shoudn't be here
    throw std::runtime_error("Unknown MemoryLocation: " +
                             std::to_string(static_cast<int>(location)));
}

}  // namespace core
}  // namespace graphdl
