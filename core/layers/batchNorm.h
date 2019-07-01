#ifndef GRAPHDL_CORE_LAYERS_BATCH_NORM_H_
#define GRAPHDL_CORE_LAYERS_BATCH_NORM_H_

#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
class BatchNormLayer : public DifferentiableLayer
{
  public:
    BatchNormLayer(ID id, const Tensor::SPtr& tensor, const Tensor::SPtr& alpha,
                   const Tensor::SPtr& beta, int numAxes);

    void initialize() override;

    ~BatchNormLayer();

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    int mNumAxes;
    Memory<float> mMean;
    Memory<float> mStddev;
};

class BatchNormGradientLayer : public Layer
{
  public:
    BatchNormGradientLayer(ID id, const Tensor::SPtr& tensor,
                           const Tensor::SPtr& alpha, const Tensor::SPtr& beta,
                           const Tensor::SPtr& out, const Tensor::SPtr& outGrad,
                           int numAxes, Memory<float> mean,
                           Memory<float> stddev);

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    int mNumAxes;
    Memory<float> mMean;
    Memory<float> mStddev;
};

void runBatchNormHost(const float* x, const float* alpha, const float* beta,
                      float* y, float* mean, float* stddev, size_t size,
                      size_t batchSize);

void runBatchNormGradientHost(const float* x, const float* alpha,
                              const float* beta, const float* y,
                              const float* yGrad, const float* mean,
                              const float* stddev, float* xGrad,
                              float* alphaGrad, float* betaGrad, size_t size,
                              size_t batchNorm);

#ifdef CUDA_AVAILABLE
namespace cuda
{
void runBatchNormDevice(const float* x, const float* alpha, const float* beta,
                        float* y, float* mean, float* stddev, size_t size,
                        size_t batchSize);

void runBatchNormGradientDevice(const float* x, const float* alpha,
                                const float* beta, const float* y,
                                const float* yGrad, const float* mean,
                                const float* stddev, float* xGrad,
                                float* alphaGrad, float* betaGrad, size_t size,
                                size_t batchNorm);
}  // namespace cuda
#endif

}  // namespace layers

Tensor::SPtr batchNorm(const Tensor::SPtr& tensor, const Tensor::SPtr& alpha,
                       const Tensor::SPtr& beta, int numAxes);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_BATCH_NORM_H_
