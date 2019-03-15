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
                const std::vector<int>& strides, PaddingType padding);

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

    void initialize() override;

    ~Conv2DLayer();

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    std::vector<int> mStrides;
    PaddingType mPadding;
    Memory<int> mGpuParams;
};

class Conv2DGradientLayer : public Layer
{
  public:
    Conv2DGradientLayer(ID id, const Tensor::SPtr& t, const Tensor::SPtr& k,
                        const Tensor::SPtr& out, const Tensor::SPtr& outG,
                        std::vector<int> strides, PaddingType padding);

    void initialize() override;

    ~Conv2DGradientLayer();

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    std::vector<int> mStrides;
    PaddingType mPadding;
    Memory<int> mGpuParams;
};

#ifdef CUDA_AVAILABLE
namespace cuda
{
extern "C" void runConv2DDevice(const float* x, const float* k, float* y,
                                size_t size, int* info, PaddingType padding);

extern "C" void runConv2DGradientDevice(const float* x, const float* k,
                                        const float* yG, float* xG, float* kG,
                                        size_t xSize, size_t kSize, int* info,
                                        PaddingType padding);

extern "C" void initializeConvGpuParams(void* dest, int* inShape, int* kerShape,
                                        int* outShape, int* strides);

}  // namespace cuda
#endif
}  // namespace layers

Tensor::SPtr convolution2D(const Tensor::SPtr& t, const Tensor::SPtr& kernel,
                           const std::vector<int>& strides,
                           layers::PaddingType padding);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_CONVOLUTION_H_
