#ifndef GRAPHDL_CORE_LAYERS_MATMUL_H_
#define GRAPHDL_CORE_LAYERS_MATMUL_H_

#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
class MatmulLayer : public DifferentiableLayer
{
  public:
    MatmulLayer(ID id, const Tensor::SPtr& m1, const Tensor::SPtr& m2);

    TensorMap gradients(Tensor::SPtr output, Tensor::SPtr outputGrad) override;

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;
};

class MatmulGradientLayer : public Layer
{
  public:
    MatmulGradientLayer(ID id, const Tensor::SPtr& m1, const Tensor::SPtr& m2,
                        const Tensor::SPtr& out, const Tensor::SPtr& outGrad);

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;
};

#ifdef CUDA_AVAILABLE
namespace cuda
{
extern "C" void runMatmulDevice(int n, int m, int k, float* X1, float* X2,
                                float* Y);

extern "C" void runMatmulGradientDevice(int n, int m, int k, float* X1,
                                        float* X2, float* Ygrad, float* X1grad,
                                        float* X2grad);

}  // namespace cuda
#endif
}  // namespace layers

Tensor::SPtr matmul(const Tensor::SPtr&, const Tensor::SPtr&);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_MATMUL_H_
