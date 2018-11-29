#ifndef GRAPHDL_CORE_INITIALIZERS_INITIALIZER_H_
#define GRAPHDL_CORE_INITIALIZERS_INITIALIZER_H_

#include "memory.h"
#include "tensorShape.h"

#include <memory>

namespace graphdl
{
namespace core
{
namespace initializers
{
class Initializer
{
  public:
    using SPtr = std::shared_ptr<Initializer>;

    Initializer(size_t seed);

    void init(float* memory, const TensorShape& shape, MemoryType type);

  protected:
    size_t mSeed;

  private:
    virtual void initHost(float* memory, const TensorShape& shape) = 0;
    virtual void initDevice(float* memory, const TensorShape& shape) = 0;
};

}  // namespace initializers
}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_INITIALIZERS_INITIALIZER_H_
