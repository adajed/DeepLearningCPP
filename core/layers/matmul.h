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
void runMatmulDevice(const float* x1, const float* x2, float* y, int n, int m,
                     int k);

void runMatmulGradientDevice(const float* x1, const float* x2,
                             const float* yGrad, float* x1Grad, float* x2Grad,
                             int n, int m, int k);

}  // namespace cuda
#endif

void runMatmulHost(const float* x1, const float* x2, float* y, int n, int m,
                   int k);

void runMatmulGradientHost(const float* x1, const float* x2, const float* yGrad,
                           float* x1Grad, float* x2Grad, int n, int m, int k);

}  // namespace layers

Tensor::SPtr matmul(const Tensor::SPtr&, const Tensor::SPtr&);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_MATMUL_H_
