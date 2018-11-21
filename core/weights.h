#ifndef GRAPHDL_CORE_WEIGHTS_H_
#define GRAPHDL_CORE_WEIGHTS_H_

#include "layer.h"

namespace graphdl
{
namespace core
{
namespace cuda
{
extern "C" void initWeights(float* memory, size_t count);
}

class WeightsLayer : public Layer
{
  public:
    WeightsLayer(ID id, const std::string& name, const Shape& shape,
                 MemoryType type);

    void initialize() override;

  private:
    void execute(const InputDict& inputs) override;
};

Tensor::SPtr weights(const std::string& name, const TensorShape& shape,
                     MemoryType type, const std::string& nspace);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_WEIGHTS_H_
