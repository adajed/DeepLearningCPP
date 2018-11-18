#include "abstractTrainer.h"

#include "abstractTensor.h"

namespace graphdl
{
namespace core
{
AbstractTrainer::AbstractTrainer(Trainer::UPtr trainer)
    : mTrainer(std::move(trainer))
{
}

ITensorPtr AbstractTrainer::optimize(const ITensorPtr& tensor) const
{
    AbstractTensor::Ptr aTensor = castITensorPtr(tensor);
    return makeAbstractTensor(mTrainer->optimize(aTensor->get()));
}

AbstractTrainer::Ptr makeAbstractTrainer(Trainer::UPtr trainer)
{
    return std::make_unique<AbstractTrainer>(std::move(trainer));
}

}  // namespace core
}  // namespace graphdl
