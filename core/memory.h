#ifndef GRAPHDL_CORE_MEMORY_H_
#define GRAPHDL_CORE_MEMORY_H_

#include "graphdl.h"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <stdexcept>

#ifdef CUDA_AVAILABLE
#include <cuda.h>
#include <cuda_runtime_api.h>

#define CUDA_CALL(op)                                          \
    {                                                          \
        cudaError_t error__ = (op);                            \
        if (error__)                                           \
            printf("Cuda error: " #op " returned op \"%s\"\n", \
                   cudaGetErrorString(error__));               \
    }

#endif

namespace graphdl
{
namespace core
{
enum class MemoryType
{
    kHOST_MEMORY = 0,
    kDEVICE_MEMORY = 1
};

template <typename T>
class Memory
{
  public:
    Memory(MemoryType type, std::size_t count)
        : mType(type), mValues(nullptr), mCount(count)
    {
    }

    MemoryType getType() const { return mType; }

    T* getValues() { return mValues; }

    const T* getValues() const { return mValues; }

    size_t getCount() const { return mCount; }

    //! \fillFrom
    //! \brief Fills memory from given host pointer.
    //! Copies sizeof(T) * getCount() number of bytes.
    //!
    void fillFrom(const T* memory)
    {
        assert(isAllocated());

        if (mType == MemoryType::kHOST_MEMORY)
        {
            std::memcpy(mValues, memory, sizeof(T) * getCount());
        }
        else  // mType == MemoryType::kDEVICE_MEMORY
        {
#ifdef CUDA_AVAILABLE
            CUDA_CALL(cudaMemcpy(mValues, memory, sizeof(T) * getCount(),
                                 cudaMemcpyHostToDevice));
#else
            throw std::runtime_error(
                "GPU support not implemented, please use CPU");
#endif
        }
    }

    //! \fn copyTo
    //! \brief Copies values to given pointer.
    //! Assumes that array can store all values.
    //!
    void copyTo(T* memory) const
    {
        assert(isAllocated());

        if (mType == MemoryType::kHOST_MEMORY)
        {
            std::memcpy(memory, mValues, sizeof(T) * getCount());
        }
        else  // mType == MemoryType::kDEVICE_MEMORY
        {
#ifdef CUDA_AVAILABLE
            CUDA_CALL(cudaMemcpy(memory, mValues, sizeof(T) * getCount(),
                                 cudaMemcpyDeviceToHost));
#else
            throw std::runtime_error(
                "GPU support not implemented, please use CPU");
#endif
        }
    }

    bool isAllocated() const { return mValues != nullptr; }

    bool allocate()
    {
        assert(!isAllocated());
        if (mType == MemoryType::kHOST_MEMORY)
        {
            mValues = new T[mCount];
            return true;
        }
        else  // mType == MemoryType::kDEVICE_MEMORY
        {
#ifdef CUDA_AVAILABLE
            CUDA_CALL(cudaMallocManaged((void**)&mValues, mCount * sizeof(T)));
            return true;
#else
            throw std::runtime_error(
                "GPU support not implemented, please use CPU");
#endif
        }
    }

    void free()
    {
        if (!isAllocated()) return;

        if (mType == MemoryType::kHOST_MEMORY)
        {
            delete[] mValues;
        }
        else  // mType == MemoryType::kDEVICE_MEMORY
        {
#ifdef CUDA_AVAILABLE
            // FIXME: add CUDA_CALL, currently "driver shutting down" error
            cudaFree(mValues);
#else
            throw std::runtime_error(
                "GPU support not implemented, please use CPU");
#endif
        }
    }

  private:
    MemoryType mType;
    T* mValues;
    std::size_t mCount;
};

inline MemoryType memoryLocationToType(MemoryLocation location)
{
    switch (location)
    {
    case MemoryLocation::kHOST: return MemoryType::kHOST_MEMORY;
    case MemoryLocation::kDEVICE: return MemoryType::kDEVICE_MEMORY;
    }

    // you shoudn't be here
    throw std::runtime_error("Unknown MemoryLocation: " +
                             std::to_string(static_cast<int>(location)));
}

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_MEMORY_H_
