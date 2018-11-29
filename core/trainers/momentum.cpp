#include "momentum.h"

#include "abstractTrainer.h"
#include "gradientBuilder.h"
#include "graphdl_train.h"
#include "initializers/constantInitializer.h"
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
MomentumTrainer::MomentumTrainer(float lr, float momentum)
    : mLearningRate(lr), mMomentum(momentum)
{
}

Tensor::SPtr MomentumTrainer::parseGradients(
    const GradientBuilder::TensorMap& grads) const
{
    GradientBuilder::TensorMap steps;
    for (const auto& grad : grads)
    {
        Tensor::SPtr w = grad.first;
        Tensor::SPtr wStep =
            weights("", w->getShape(), constantInitializer(0.), w->getType(),
                    core::TRAIN_WEIGHTS_NAMESPACE);
        steps.insert({w, wStep});
    }

    std::vector<Tensor::SPtr> stepUpdates;
    stepUpdates.reserve(grads.size());
    for (const auto& grad : grads)
    {
        Tensor::SPtr w = grad.first;
        Tensor::SPtr g = grad.second;
        Tensor::SPtr s = steps[w];

        Tensor::SPtr a = assign(s, mMomentum * s + mLearningRate * g);
        stepUpdates.push_back(a);
    }

    std::vector<Tensor::SPtr> weightsUpdates;
    weightsUpdates.reserve(grads.size());
    for (const auto& step : steps)
    {
        Tensor::SPtr w = step.first;
        Tensor::SPtr s = step.second;

        Tensor::SPtr a = assign(w, w - s);
        weightsUpdates.push_back(a);
    }

    return queue({group(stepUpdates), group(weightsUpdates)});
}

}  // namespace trainers
}  // namespace core

namespace train
{
ITrainerPtr momentum(float lr, float m)
{
    core::Trainer::UPtr t =
        std::make_unique<core::trainers::MomentumTrainer>(lr, m);
    return core::makeAbstractTrainer(std::move(t));
}

}  // namespace train
}  // namespace graphdl
