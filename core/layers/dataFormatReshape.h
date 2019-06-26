#ifndef GRAPHDL_CORE_LAYERS_DATA_FORMAT_RESHAPE_H_
#define GRAPHDL_CORE_LAYERS_DATA_FORMAT_RESHAPE_H_

#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
class Nhwc2NchwLayer : public DifferentiableLayer
{
  public:
    Nhwc2NchwLayer(ID id, const Tensor::SPtr& tensor);

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;
};

class Nchw2NhwcLayer : public DifferentiableLayer
{
  public:
    Nchw2NhwcLayer(ID id, const Tensor::SPtr& tensor);

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;
};

#ifdef CUDA_AVAILABLE
namespace cuda
{
void runNhwc2NchwDevice(const float* in, float* out, int* outShape);

void runNchw2NhwcDevice(const float* in, float* out, int* outShape);

}  // namespace cuda
#endif

void runNhwc2NchwHost(const float* in, float* out, const int* outShape);

void runNchw2NhwcHost(const float* in, float* out, const int* outShape);

}  // namespace layers

Tensor::SPtr nhwc2nchw(const Tensor::SPtr& tensor);

Tensor::SPtr nchw2nhwc(const Tensor::SPtr& tensor);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_DATA_FORMAT_RESHAPE_H_
