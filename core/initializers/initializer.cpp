#include "initializer.h"

#include <random>

namespace graphdl
{
namespace core
{
namespace initializers
{
Initializer::Initializer(size_t seed) : mSeed(seed) {}

void Initializer::init(float* memory, const TensorShape& shape, MemoryType type)
{
    if (type == MemoryType::kHOST_MEMORY)
        initHost(memory, shape);
    else  // type == MemoryType::kDEVICE_MEMORY
        initDevice(memory, shape);

    // produce next seed
    mSeed = std::mt19937(mSeed)();
}

}  // namespace initializers
}  // namespace core
}  // namespace graphdl
