#ifndef GRAPHDL_CORE_LAYERS_REDUCE_SUM_H_
#define GRAPHDL_CORE_LAYERS_REDUCE_SUM_H_

#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
class ReduceSumLayer : public DifferentiableLayer
{
  public:
    ReduceSumLayer(ID id, Tensor::SPtr tensor);

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

  private:
    void execute(const InputDict& inputs) override;
};

class ReduceSumGradientLayer : public Layer
{
  public:
    ReduceSumGradientLayer(ID id, Tensor::SPtr in, Tensor::SPtr out,
                           Tensor::SPtr outGrad);

  private:
    void execute(const InputDict& inputs) override;
};

}  // namespace layers

Tensor::SPtr reduceSum(Tensor::SPtr tensor);

}  // namespace core

}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_REDUCE_SUM_H_
