#include "trainer.h"

#include "gradientBuilder.h"
#include "graph.h"

namespace graphdl
{
namespace core
{
Tensor::SPtr Trainer::optimize(const Tensor::SPtr& tensor) const
{
    std::vector<Tensor::SPtr> weights;
    weights.reserve(getDefaultGraph()->getWeights().size());
    for (auto pair : core::getDefaultGraph()->getWeights())
        weights.push_back(pair.second);
    GradientBuilder builder(tensor, weights);
    GradientBuilder::TensorMap grads = builder.createGradients();

    return parseGradients(grads);
}

}  // namespace core
}  // namespace graphdl
