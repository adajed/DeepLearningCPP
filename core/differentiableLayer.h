#ifndef GRAPHDL_CORE_DIFFERENTIABLE_LAYER_H_
#define GRAPHDL_CORE_DIFFERENTIABLE_LAYER_H_

#include "layer.h"

namespace graphdl
{
namespace core
{
class DifferentiableLayer : public Layer
{
   public:
    DifferentiableLayer(ID id, const std::vector<Tensor::SPtr>& inputs,
                        std::vector<Tensor::SPtr> outputs)
        : Layer(id, inputs, outputs)
    {
    }

    //! \fn gradients
    virtual TensorMap gradients(Tensor::SPtr output,
                                Tensor::SPtr outputGrad) override = 0;

    bool hasGradient() const override final { return true; }
};

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_DIFFERENTIABLE_LAYER_H_
