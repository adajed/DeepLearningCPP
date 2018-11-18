#ifndef GRAPHDL_CORE_ABSTRACT_TRAINER_H_
#define GRAPHDL_CORE_ABSTRACT_TRAINER_H_

#include "graphdl_train.h"
#include "trainer.h"

namespace graphdl
{
namespace core
{
class AbstractTrainer : public train::ITrainer
{
  public:
    using Ptr = std::unique_ptr<AbstractTrainer>;

    AbstractTrainer(Trainer::UPtr trainer);

    ITensorPtr optimize(const ITensorPtr& tensor) const override;

  private:
    Trainer::UPtr mTrainer;
};

AbstractTrainer::Ptr makeAbstractTrainer(Trainer::UPtr trainer);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_ABSTRACT_TRAINER_H_
