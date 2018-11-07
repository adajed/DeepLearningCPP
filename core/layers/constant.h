#ifndef GRAPHDL_CORE_CONSTANT_LAYER_H_
#define GRAPHDL_CORE_CONSTANT_LAYER_H_

#include "layer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
class ConstantLayer : public Layer
{
  public:
    ConstantLayer(ID id, float value, const TensorShape& shape,
                  MemoryType type);

    void initialize() override;

  private:
    void execute(const InputDict& inputs) override;

    float mValue;
};

#ifdef CUDA_AVAILABLE
namespace cuda
{
void fillWithValue(std::size_t size, float* x, float val);

}  // namespace cuda
#endif
}  // namespace layers

Tensor::SPtr constant(float value, const TensorShape& shape, MemoryType type);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_CONSTANT_LAYER_H_
