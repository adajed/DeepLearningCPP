#include "initializer.h"

namespace graphdl
{
namespace core
{
namespace initializers
{
void Initializer::init(float* memory, const TensorShape& shape,
                       MemoryType type) const
{
    if (type == MemoryType::kHOST_MEMORY)
        initHost(memory, shape);
    else  // type == MemoryType::kDEVICE_MEMORY
        initDevice(memory, shape);
}

}  // namespace initializers
}  // namespace core
}  // namespace graphdl
