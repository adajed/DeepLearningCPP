#ifndef GRAPHDL_CORE_LAYERS_SOFTMAX_H_
#define GRAPHDL_CORE_LAYERS_SOFTMAX_H_

#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
class SoftmaxLayer : public DifferentiableLayer
{
  public:
    SoftmaxLayer(ID id, const Tensor::SPtr& x, int numAxes);

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

    void initialize() override;

    ~SoftmaxLayer();

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    int mNumAxes;
    Memory<float> mWorkingSpace;
    size_t mOutSize;
    size_t mReduceSize;
};

class SoftmaxGradientLayer : public Layer
{
  public:
    SoftmaxGradientLayer(ID id, const Tensor::SPtr& x, int numAxes,
                         Tensor::SPtr y, Tensor::SPtr yGrad);

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    int mNumAxes;
    size_t mOutSize;
    size_t mReduceSize;
};

#ifdef CUDA_AVAILABLE
namespace cuda
{
void runSoftmaxDevice(const float* x, float* w, float* y, size_t outSize,
                      size_t reduceSize);

void runSoftmaxGradientDevice(const float* x, const float* y,
                              const float* yGrad, float* xGrad, size_t outSize,
                              size_t reduceSize);

}  // namespace cuda
#endif

}  // namespace layers

Tensor::SPtr softmax(const Tensor::SPtr& tensor, int numAxes);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_SOFTMAX_H_
