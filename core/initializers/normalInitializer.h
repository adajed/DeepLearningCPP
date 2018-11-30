#ifndef GRAPHDL_CORE_INITIALIZERS_NORMAL_H_
#define GRAPHDL_CORE_INITIALIZERS_NORMAL_H_

#include "initializer.h"

namespace graphdl
{
namespace core
{
namespace initializers
{
class NormalInitializer : public Initializer
{
  public:
    NormalInitializer(float mean, float stddev, size_t seed);

  private:
    void initHost(float* memory, const TensorShape& shape) override;

    void initDevice(float* memory, const TensorShape& shape) override;

    float mMean, mStddev;
    size_t mSeed{};
};

}  // namespace initializers
}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_INITIALIZERS_NORMAL_H_
