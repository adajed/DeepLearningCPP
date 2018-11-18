#ifndef GRAPHDL_CORE_TRAINER_H_
#define GRAPHDL_CORE_TRAINER_H_

#include "gradientBuilder.h"
#include "layer.h"

namespace graphdl
{
namespace core
{
class Trainer
{
  public:
    using UPtr = std::unique_ptr<Trainer>;

    Tensor::SPtr optimize(const Tensor::SPtr& tensor) const;

    virtual ~Trainer() {}

  private:
    virtual Tensor::SPtr parseGradients(
        const GradientBuilder::TensorMap& grads) const = 0;
};

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_TRAINER_H_
