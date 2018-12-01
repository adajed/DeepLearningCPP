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
    kSAME  = 1
};

class Pooling2DLayer : public DifferentiableLayer
{
  public:
    Pooling2DLayer(ID id, const Tensor::SPtr& t,
                 PoolingType pooling,
                 const std::vector<int>& kernel,
                 const std::vector<int>& strides,
                 PaddingType padding);

    TensorMap gradients(Tensor::SPtr output, Tensor::SPtr outputGrad) override;

  private:
    void execute(const InputDict& inputs) override;

    PoolingType mPooling;
    std::vector<int> mKernelWindow;
    std::vector<int> mStrides;
    PaddingType mPadding;
};

class Pooling2DGradientLayer : public Layer
{
  public:
    PoolingGradientLayer(ID id, const Tensor::SPtr& t,
                         const Tensor::SPtr out,
                         const Tensor::SPtr& outGrad,
                         PoolingType pooling,
                         const std::vector<int>& kernel,
                         const std::vector<int>& strides,
                         PaddingType padding);

  private:
    void execute(const InputDict& inputs) override;

    PoolingType mPooling;
    std::vector<int> mKernelWindow;
    std::vector<int> mStrides;
    PaddingType mPadding;
};

}  // namespace layers

Tensor::SPtr pooling2D(const Tensor::SPtr& t,
                       layers::PoolingType pooling,
                       const std::vector<int>& kernel,
                       const std::vector<int>& strides,
                       layers::PaddingType padding);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_POOLING_H_
