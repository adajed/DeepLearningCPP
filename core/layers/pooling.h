#ifndef GRAPHDL_CORE_LAYERS_POOLING_H_
#define GRAPHDL_CORE_LAYERS_POOLING_H_

#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
enum class PoolingType
{
    kMAX = 0,
    kAVERAGE = 1
};

enum class PaddingType
{
    kVALID = 0,
    kSAME = 1
};

class Pooling2DLayer : public DifferentiableLayer
{
  public:
    Pooling2DLayer(ID id, const Tensor::SPtr& t, PoolingType pooling,
                   const std::vector<int>& kernel,
                   const std::vector<int>& strides, PaddingType padding);

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

    void initialize() override;

    ~Pooling2DLayer() override;

  private:
    void execute(const InputDict& inputs) override;

    PoolingType mPooling;
    std::vector<int> mKernelWindow;
    std::vector<int> mStrides;
    PaddingType mPadding;
    Memory<int> mGpuParams;
};

class Pooling2DGradientLayer : public Layer
{
  public:
    Pooling2DGradientLayer(ID id, const Tensor::SPtr& t,
                           const Tensor::SPtr& out, const Tensor::SPtr& outGrad,
                           PoolingType pooling, std::vector<int> kernel,
                           std::vector<int> strides, PaddingType padding);

    void initialize() override;

    ~Pooling2DGradientLayer() override;

  private:
    void execute(const InputDict& inputs) override;

    PoolingType mPooling;
    std::vector<int> mKernelWindow;
    std::vector<int> mStrides;
    PaddingType mPadding;
    Memory<int> mGpuParams;
};

#ifdef CUDA_AVAILABLE
namespace cuda
{
extern "C" void runPool2DDevice(const float* x, float* y, int* info,
                                size_t size, PoolingType pooling,
                                PaddingType padding);

extern "C" void runPool2DGradientDevice(const float* x, const float* y,
                                        const float* yG, float* xG, int* info,
                                        size_t size, PoolingType pooling,
                                        PaddingType padding);

extern "C" void initializePoolGpuParams(void* dest, int* inShape, int* kernel,
                                        int* strides, int* outShape);

}  // namespace cuda
#endif

}  // namespace layers

Tensor::SPtr pooling2D(const Tensor::SPtr& t, layers::PoolingType pooling,
                       const std::vector<int>& kernel,
                       const std::vector<int>& strides,
                       layers::PaddingType padding);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_POOLING_H_
