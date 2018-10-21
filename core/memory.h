#ifndef DLL_CORE_MEMORY_H_
#define DLL_CORE_MEMORY_H_

#include <cstddef>

#include "dll.h"

namespace dll
{
namespace core
{

enum class MemoryType
{
    kHOST_MEMORY = 0,
    kDEVICE_MEMORY = 1
};

class Memory
{
public:
    Memory(MemoryType type, size_t count);

    bool isAllocated() const;

    float* getValues();

    const float* getValues() const;

    std::size_t getCount() const;

    void fill(HostTensor* hostTensor) const;

    bool allocate();

    void free();


private:
    MemoryType mType;
    float* mValues;
    std::size_t mCount;
};

} // namespace core
} // namespace dll

#endif // DLL_CORE_MEMORY_H_
