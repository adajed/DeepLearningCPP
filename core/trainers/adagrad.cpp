#include "adagrad.h"

#include "abstractTrainer.h"
#include "graphdl_train.h"
#include "initializers/constantInitializer.h"
#include "layers/activation.h"
#include "layers/assign.h"
#include "layers/constant.h"
#include "layers/elementwise.h"
#include "layers/group.h"
#include "layers/queue.h"
#include "weights.h"
#include "weightsNamespaces.h"

namespace graphdl
{
namespace core
{
namespace trainers
{
AdagradTrainer::AdagradTrainer(float lr, float eps)
    : mLearningRate(lr), mEpsilon(eps)
{
}

Tensor::SPtr AdagradTrainer::parseGradients(
    const GradientBuilder::TensorMap& grads) const
{
    std::vector<Tensor::SPtr> gradUpdates;
    std::vector<Tensor::SPtr> weightsUpdates;
    for (const auto& grad : grads)
    {
        Tensor::SPtr w = grad.first;
        Tensor::SPtr g = grad.second;
        Tensor::SPtr gSquared =
            weights("", w->getShape(), constantInitializer(0.), w->getType(),
                    core::TRAIN_WEIGHTS_NAMESPACE);

        gradUpdates.push_back(assign(gSquared, gSquared + square(g)));
        Tensor::SPtr delta = (mLearningRate / sqrt(gSquared + mEpsilon)) * g;
        weightsUpdates.push_back(assign(w, w - delta));
    }

    return queue({group(gradUpdates), group(weightsUpdates)});
}

}  // namespace trainers
}  // namespace core

namespace train
{
ITrainerPtr adagrad(float lr, float eps)
{
    core::Trainer::UPtr t =
        std::make_unique<core::trainers::AdagradTrainer>(lr, eps);
    return core::makeAbstractTrainer(std::move(t));
}
}  // namespace train
}  // namespace graphdl
