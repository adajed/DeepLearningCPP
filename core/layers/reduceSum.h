#ifndef GRAPHDL_CORE_LAYERS_REDUCE_SUM_H_
#define GRAPHDL_CORE_LAYERS_REDUCE_SUM_H_

#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
class ReduceSumLayer : public DifferentiableLayer
{
  public:
    ReduceSumLayer(ID id, const Tensor::SPtr& tensor, int numAxes);

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

  private:
    void execute(const InputDict& inputs) override;

    int mNumAxes;
};

class ReduceSumGradientLayer : public Layer
{
  public:
    ReduceSumGradientLayer(ID id, const Tensor::SPtr& in, int numAxes,
                           Tensor::SPtr out, Tensor::SPtr outGrad);

  private:
    void execute(const InputDict& inputs) override;

    int mNumAxes;
};

#ifdef CUDA_AVAILABLE
namespace cuda
{
void runReduceSumDevice(const float* x, float* y, size_t outSize,
                        size_t reduceSize);

void runReduceSumGradientDevice(const float* yGrad, float* xGrad,
                                size_t outSize, size_t reduceSize);

}  // namespace cuda
#endif

void runReduceSumHost(const float* x, float* y, size_t outSize,
                      size_t reduceSize);

void runReduceSumGradientHost(const float* yGrad, float* xGrad, size_t outSize,
                              size_t reduceSize);

}  // namespace layers

Tensor::SPtr reduceSum(Tensor::SPtr t, int numAxes);

Tensor::SPtr reduceMean(const Tensor::SPtr& t, int numAxes);

}  // namespace core

}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_REDUCE_SUM_H_
