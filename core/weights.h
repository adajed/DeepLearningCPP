#ifndef GRAPHDL_CORE_WEIGHTS_H_
#define GRAPHDL_CORE_WEIGHTS_H_

#include "initializers/initializer.h"
#include "layer.h"

namespace graphdl
{
namespace core
{
namespace cuda
{
extern "C" void initWeights(float* memory, size_t count);
}

using namespace initializers;

class WeightsLayer : public Layer
{
  public:
    WeightsLayer(ID id, const std::string& name, const Shape& shape,
                 Initializer::SPtr initializer, MemoryType type);

    void initialize() override;

  private:
    void execute(const InputDict& inputs) override;

    Initializer::SPtr mInitializer;
};

Tensor::SPtr weights(const std::string& name, const TensorShape& shape,
                     Initializer::SPtr initializer, MemoryType type,
                     const std::string& nspace);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_WEIGHTS_H_
