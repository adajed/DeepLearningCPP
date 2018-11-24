#ifndef GRAPHDL_CORE_TRAINERS_ADAGRAD_H_
#define GRAPHDL_CORE_TRAINERS_ADAGRAD_H_

#include "trainer.h"

namespace graphdl
{
namespace core
{
namespace trainers
{
class AdagradTrainer : public Trainer
{
  public:
    AdagradTrainer(float lr, float eps);

  private:
    Tensor::SPtr parseGradients(
        const GradientBuilder::TensorMap& grads) const override;

    float mLearningRate;
    float mEpsilon;
};

}
}
}

#endif  // GRAPHDL_CORE_TRAINERS_ADAGRAD_H_
