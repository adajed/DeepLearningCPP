#ifndef GRAPHDL_CORE_INITIALIZERS_CONSTANT_H_
#define GRAPHDL_CORE_INITIALIZERS_CONSTANT_H_

#include "initializer.h"

namespace graphdl
{
namespace core
{
namespace initializers
{
class ConstantInitializer : public Initializer
{
  public:
    ConstantInitializer(float value);

  private:
    void initHost(float* memory, const TensorShape& shape) const override;

    void initDevice(float* memory, const TensorShape& shape) const override;

    float mValue;
};

}  // namespace initializers
}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_INITIALIZERS_CONSTANT_H_
