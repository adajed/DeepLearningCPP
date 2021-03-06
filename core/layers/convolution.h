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
    Conv2DLayer(ID id, const Tensor::SPtr& t, const Tensor::SPtr& kernel,
                const std::vector<int>& strides, PaddingType padding,
                DataFormat dataFormat);

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

    void initialize() override;

    ~Conv2DLayer();

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    std::vector<int> mStrides;
    PaddingType mPadding;
    DataFormat mDataFormat;
    Memory<int> mGpuParams;
};

class Conv2DGradientLayer : public Layer
{
  public:
    Conv2DGradientLayer(ID id, const Tensor::SPtr& t, const Tensor::SPtr& k,
                        const Tensor::SPtr& out, const Tensor::SPtr& outG,
                        std::vector<int> strides, PaddingType padding,
                        DataFormat dataFormat);

    void initialize() override;

    ~Conv2DGradientLayer();

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    std::vector<int> mStrides;
    PaddingType mPadding;
    DataFormat mDataFormat;
    Memory<int> mGpuParams;
};

#ifdef CUDA_AVAILABLE
namespace cuda
{
void runConv2DDevice(const float* x, const float* k, float* y,
                     const int* params, PaddingType padding,
                     DataFormat dataFormat);

void runConv2DGradientDevice(const float* x, const float* k, const float* yG,
                             float* xG, float* kG, const int* info,
                             PaddingType padding, DataFormat dataFormat);

}  // namespace cuda
#endif
}  // namespace layers

Tensor::SPtr convolution2D(const Tensor::SPtr& t, const Tensor::SPtr& kernel,
                           const std::vector<int>& strides,
                           layers::PaddingType padding,
                           layers::DataFormat dataFormat);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_CONVOLUTION_H_
