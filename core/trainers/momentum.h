#ifndef GRAPHDL_CORE_TRAINERS_MOMENTUM_H_
#define GRAPHDL_CORE_TRAINERS_MOMENTUM_H_

#include "trainer.h"

namespace graphdl
{
namespace core
{
namespace trainers
{
class MomentumTrainer : public Trainer
{
  public:
    MomentumTrainer(float lr, float momentum);

  private:
    Tensor::SPtr parseGradients(
        const GradientBuilder::TensorMap& grads) const override;

    float mLearningRate;
    float mMomentum;
};

}  // namespace trainers
}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_TRAINERS_MOMENTUM_H_
