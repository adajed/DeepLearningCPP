#ifndef GRAPHDL_CORE_LAYERS_RESHAPE_H_
#define GRAPHDL_CORE_LAYERS_RESHAPE_H_

#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
class ReshapeLayer : public DifferentiableLayer
{
  public:
    ReshapeLayer(ID id, const Tensor::SPtr& t, const TensorShape& shape);

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;
};

}  // namespace layers

Tensor::SPtr reshape(const Tensor::SPtr& t, const TensorShape& shape);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_RESHAPE_H_
