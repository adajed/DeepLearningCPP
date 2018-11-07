#ifndef GRAPHDL_CORE_MEMORY_H_
#define GRAPHDL_CORE_MEMORY_H_

#include <cstddef>

namespace graphdl
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
    Memory(MemoryType type, std::size_t count);

    float* getValues();
    const float* getValues() const;

    float& operator[](std::size_t pos);
    const float& operator[](std::size_t pos) const;

    std::size_t getCount() const;

    //! \fn fill
    //! \brief Copies values to given pointer.
    //! Assumes that array can store all values.
    //!
    void fill(float* memory) const;

    bool isAllocated() const;

    bool allocate();

    void free();

  private:
    MemoryType mType;
    float* mValues;
    std::size_t mCount;
};

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_MEMORY_H_
