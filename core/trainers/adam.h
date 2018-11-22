#ifndef GRAPHDL_CORE_TRAINERS_ADAM_H_
#define GRAPHDL_CORE_TRAINERS_ADAM_H_

#include "trainer.h"

namespace graphdl
{
namespace core
{
namespace trainers
{
class AdamTrainer : public Trainer
{
  public:
    AdamTrainer(float lr, float beta1, float beta2, float eps);

  private:
    Tensor::SPtr parseGradients(
        const GradientBuilder::TensorMap& grads) const override;

    float mLearningRate;
    float mBeta1, mBeta2;
    float mEpsilon;
};

}  // namespace trainers
}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_TRAINERS_ADAM_H_
