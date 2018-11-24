#include "gradientDescent.h"

#include "abstractTrainer.h"
#include "graph.h"
#include "graphdl_train.h"
#include "layers/assign.h"
#include "layers/constant.h"
#include "layers/elementwise.h"
#include "layers/group.h"

namespace graphdl
{
namespace core
{
namespace trainers
{
GradientDescentTrainer::GradientDescentTrainer(float lr) : mLearningRate(lr) {}

Tensor::SPtr GradientDescentTrainer::parseGradients(
    const GradientBuilder::TensorMap& grads) const
{
    std::vector<Tensor::SPtr> modifications;
    modifications.reserve(grads.size());
    for (const auto& grad : grads)
    {
        Tensor::SPtr w = grad.first;
        Tensor::SPtr g = grad.second;
        Tensor::SPtr a = assign(w, w - mLearningRate * g);
        modifications.push_back(a);
    }

    return group(modifications);
}

}  // namespace trainers
}  // namespace core

namespace train
{
ITrainerPtr gradientDescent(float lr)
{
    core::Trainer::UPtr t =
        std::make_unique<core::trainers::GradientDescentTrainer>(lr);
    return core::makeAbstractTrainer(std::move(t));
}

}  // namespace train
}  // namespace graphdl
