#ifndef GRAPHDL_CORE_LAYERS_CONVOLUTION_H_
#define GRAPHDL_CORE_LAYERS_CONVOLUTION_H_

#include "differentiableLayer.h"
#include "pooling.h"

namespace graphdl
{
namespace core
{
namespace layers
{

class Conv2DLayer : public DifferentiableLayer
{
  public:
    Conv2DLayer(ID id, const Tensor::SPtr& t,
                const Tensor::SPtr& kernel,
                const std::vector<int>& strides,
                PaddingType padding);

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

  private:
    void execute(const InputDict& inputs) override;

    std::vector<int> mStrides;
    PaddingType mPadding;
};

class Conv2DGradientLayer : public Layer
{
  public:
    Conv2DGradientLayer(ID id, const Tensor::SPtr& t,
                        const Tensor::SPtr& kernel,
                        const Tensor::SPtr& out,
                        const Tensor::SPtr& outGrad,
                        const std::vector<int>& strides,
                        PaddingType padding);

  private:
    void execute(const InputDict& inputs) override;

    std::vector<int> mStrides;
    PaddingType mPadding;
};

#ifdef CUDA_AVAILABLE
namespace cuda
{
extern "C" void runConv2DDevice(
        const float* x, const float* k, float* y,
        int* shape, int cOut, int* strides, PaddingType padding);

extern "C" void runConv2DGradientDevice(
        const float* x, const float* k, const float* y, const float* yG,
        float* xG, float* kG, int* shape, int cOut, int* strides,
        PaddingType padding);

}  // namespace cuda
#endif
}  // namespace layers

Tensor::SPtr convolution2D(const Tensor::SPtr& t,
                           const Tensor::SPtr& kernel,
                           const std::vector<int>& strides,
                           layers::PaddingType padding);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_CONVOLUTION_H_
