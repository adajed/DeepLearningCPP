#include "momentum.h"

#include "abstractTrainer.h"
#include "gradientBuilder.h"
#include "graphdl_train.h"
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
        /* Tensor::SPtr wStep = weights("", w->getShape(), w->getType(),
         * core::TRAIN_WEIGHTS_NAMESPACE); */
        Tensor::SPtr wStep = constant(0., w->getShape(), w->getType());
        steps.insert({w, wStep});
    }

    std::vector<Tensor::SPtr> stepUpdates;
    stepUpdates.reserve(grads.size());
    for (const auto& grad : grads)
    {
        Tensor::SPtr w = grad.first;
        Tensor::SPtr g = grad.second;
        Tensor::SPtr s = steps[w];
        Tensor::SPtr m = constant(mMomentum, {}, w->getType());
        Tensor::SPtr lr = constant(mLearningRate, {}, w->getType());

        Tensor::SPtr a = assign(s, m * s + lr * g);
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
