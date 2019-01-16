#ifndef GRAPHDL_CORE_INITIALIZERS_UNIFORM_H_
#define GRAPHDL_CORE_INITIALIZERS_UNIFORM_H_

#include "initializer.h"

namespace graphdl
{
namespace core
{
namespace initializers
{
class UniformInitializer : public Initializer
{
  public:
    UniformInitializer(float min, float max, size_t seed);

  private:
    void initHost(float* memory, const TensorShape& shape) override;

    void initDevice(float* memory, const TensorShape& shape) override;

    float mMinValue, mMaxValue;
    size_t mSeed{};
};

}  // namespace initializers
}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_INITIALIZERS_UNIFORM_H_
