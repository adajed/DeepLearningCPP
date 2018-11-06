#ifndef GRAPHDL_CORE_LAYERS_MATMUL_H_
#define GRAPHDL_CORE_LAYERS_MATMUL_H_

#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
class MatmulLayer : public DifferentiableLayer
{
   public:
    MatmulLayer(ID id, const Tensor::SPtr& m1, const Tensor::SPtr& m2);

    TensorMap gradients(Tensor::SPtr output, Tensor::SPtr outputGrad) override;

   private:
    void execute(const InputDict& inputs) override;
};

class MatmulGradientLayer : public Layer
{
   public:
    MatmulGradientLayer(ID id, const Tensor::SPtr& m1, const Tensor::SPtr& m2,
                        const Tensor::SPtr& out, const Tensor::SPtr& outGrad);

   private:
    void execute(const InputDict&) override;
};

}  // namespace layers

Tensor::SPtr matmul(const Tensor::SPtr&, const Tensor::SPtr&);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_MATMUL_H_
