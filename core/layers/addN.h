#ifndef GRAPHDL_CORE_LAYERS_ADDN_LAYER_H_
#define GRAPHDL_CORE_LAYERS_ADDN_LAYER_H_

#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
class AddNLayer : public DifferentiableLayer
{
  public:
    AddNLayer(ID id, std::vector<Tensor::SPtr> tensors);

    DifferentiableLayer::TensorMap gradients(Tensor::SPtr out,
                                             Tensor::SPtr outGrad) override;

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;
};

class AddNGradientLayer : public Layer
{
  public:
    AddNGradientLayer(ID id, const std::vector<Tensor::SPtr>& ins,
                      const Tensor::SPtr& out, const Tensor::SPtr& outGrad);

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;
};

#ifdef CUDA_AVAILABLE
namespace cuda
{
void runAddNDevice(int n, size_t size, float** xs, float* y);

void runAddNGradientDevice(int n, size_t size, float* yGrad, float** xGrads);

}  // namespace cuda
#endif

void runAddNHost(float** xs, int n, float* y, size_t size);

void runAddNGradientHost(const float* yGrad, float** xGrads, int n,
                         size_t size);

}  // namespace layers

Tensor::SPtr addN(std::vector<Tensor::SPtr> tensors);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_ADDN_LAYER_H_
