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

    GradientBuilder(const Tensor::SPtr& tensor);

    TensorMap createGradients();

  private:
    void findTensorOutputs(const Tensor::SPtr& tensor,
                           std::set<Tensor::SPtr>& visited);
    void modifyTensorGradient(Tensor::SPtr tensor,
                              const Tensor::SPtr& tensorGrad);
    void calculateGradientsForTensor(const Tensor::SPtr& tensor);

    Tensor::SPtr mTensor;
    std::map<Tensor::SPtr, std::vector<Tensor::SPtr>> mTensorGradients;
    std::map<Tensor::SPtr, std::set<Tensor::SPtr>> mGradientsToCalc;
    std::map<Tensor::SPtr, Tensor::SPtr> mCalculatedTensors;
};

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_GRADIENT_BUILDER_H_
