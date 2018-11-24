#include "adam.h"

#include "abstractTrainer.h"
#include "gradientBuilder.h"
#include "graphdl_train.h"
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
AdamTrainer::AdamTrainer(float lr, float beta1, float beta2, float eps)
    : mLearningRate(lr), mBeta1(beta1), mBeta2(beta2), mEpsilon(eps)
{
}

Tensor::SPtr AdamTrainer::parseGradients(
    const GradientBuilder::TensorMap& grads) const
{
    GradientBuilder::TensorMap wM;
    GradientBuilder::TensorMap wV;
    std::vector<Tensor::SPtr> updatesM;
    std::vector<Tensor::SPtr> updatesV;
    std::vector<Tensor::SPtr> correct;
    std::vector<Tensor::SPtr> updates;
    for (const auto& grad : grads)
    {
        Tensor::SPtr w = grad.first;
        Tensor::SPtr g = grad.second;
        Tensor::SPtr m = constant(0., w->getShape(), w->getType());
        Tensor::SPtr v = constant(0., w->getShape(), w->getType());

        wM.insert({w, m});
        wV.insert({w, v});

        updatesM.push_back(assign(m, mBeta1 * m + (1. - mBeta1) * g));
        updatesV.push_back(assign(v, mBeta1 * v + (1. - mBeta2) * square(g)));

        Tensor::SPtr b1_step = constant(1., {}, w->getType());
        Tensor::SPtr b2_step = constant(1., {}, w->getType());
        correct.push_back(assign(b1_step, b1_step * mBeta1));
        correct.push_back(assign(b2_step, b2_step * mBeta2));

        Tensor::SPtr m_dash = m / (1. - b1_step);
        Tensor::SPtr v_dash = v / (1. - b2_step);
        Tensor::SPtr delta = mLearningRate / (sqrt(v_dash) + mEpsilon) * m_dash;
        Tensor::SPtr a = assign(w, w - delta);
        updates.push_back(a);
    }

    return queue(
        {group(updatesM), group(updatesV), group(correct), group(updates)});
}

}  // namespace trainers
}  // namespace core

namespace train
{
ITrainerPtr adam(float lr, float beta1, float beta2, float eps)
{
    core::Trainer::UPtr t =
        std::make_unique<core::trainers::AdamTrainer>(lr, beta1, beta2, eps);
    return core::makeAbstractTrainer(std::move(t));
}

}  // namespace train
}  // namespace graphdl
