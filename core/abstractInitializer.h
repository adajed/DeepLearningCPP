#ifndef GRAPHDL_CORE_ABSTRACT_INITIALIZER_H_
#define GRAPHDL_CORE_ABSTRACT_INITIALIZER_H_

#include "graphdl.h"
#include "initializers/initializer.h"

namespace graphdl
{
namespace core
{
using namespace initializers;

class AbstractInitializer : public IInitializer
{
  public:
    AbstractInitializer(Initializer::SPtr initializer);

    void init(float* memory, const Shape& shape,
              MemoryLocation location) const override;

    Initializer::SPtr get();

  private:
    Initializer::SPtr mInitializer;
};

SharedPtr<AbstractInitializer> castIInitializer(
    const SharedPtr<IInitializer>& initializer);

SharedPtr<AbstractInitializer> makeAbstractInitializer(
    Initializer::SPtr initializer);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_ABSTRACT_INITIALIZER_H_
