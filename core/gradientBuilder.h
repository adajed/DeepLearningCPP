#ifndef GRAPHDL_CORE_GRADIENT_BUILDER_H_
#define GRAPHDL_CORE_GRADIENT_BUILDER_H_

#include "layer.h"

#include <set>

namespace graphdl
{
namespace core
{
class GradientBuilder
{
  public:
    using TensorMap = std::map<Tensor::SPtr, Tensor::SPtr>;

    GradientBuilder(Tensor::SPtr tensor);

    TensorMap createGradients();

  private:
    void findTensorOutputs(Tensor::SPtr tensor,
                           std::set<Tensor::SPtr>& visited);
    void modifyTensorGradient(Tensor::SPtr tensor, Tensor::SPtr tensorGrad);
    void calculateGradientsForTensor(Tensor::SPtr tensor);

    Tensor::SPtr mTensor;
    std::map<Tensor::SPtr, std::vector<Tensor::SPtr>> mTensorGradients;
    std::map<Tensor::SPtr, std::set<Tensor::SPtr>> mGradientsToCalc;
    std::map<Tensor::SPtr, Tensor::SPtr> mCalculatedTensors;
};

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_GRADIENT_BUILDER_H_
