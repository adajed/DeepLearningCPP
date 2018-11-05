#ifndef GRAPHDL_CORE_LAYERS_ELEMENTWISE_H_
#define GRAPHDL_CORE_LAYERS_ELEMENTWISE_H_

#include <functional>
#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
enum class Elementwise : int
{
    kADD = 0,
    kSUB = 1,
    kMUL = 2,
    kDIV = 3
};

using ElementwiseFun = std::function<float(float, float)>;

class ElementwiseLayer : public DifferentiableLayer
{
   public:
    ElementwiseLayer(ID id, Tensor::SPtr t1, Tensor::SPtr t2, Elementwise op);

    TensorMap gradients(Tensor::SPtr output, Tensor::SPtr outputGrad) override;

   private:
    void execute(const InputDict& inputs) override;

    Elementwise mOp;
    ElementwiseFun mFun;
};

class ElementwiseGradientLayer : public Layer
{
   public:
    ElementwiseGradientLayer(ID id, Tensor::SPtr t1, Tensor::SPtr t2,
                             Tensor::SPtr out, Tensor::SPtr outGrad,
                             Elementwise op);

   private:
    static std::vector<Tensor::SPtr> createOutputs(Tensor::SPtr, Tensor::SPtr);

    void execute(const InputDict&) override;

    Elementwise mOp;
    ElementwiseFun mFun1, mFun2;
};

#ifdef CUDA_AVAILABLE
namespace cuda
{
void runElementwiseDevice(std::size_t size, float* x1, float* x2, float* y,
                          Elementwise op);

void runElementwiseGradientDevice(std::size_t size, float* x1, float* x2,
                                  float* yG, float* x1G, float* x2G,
                                  Elementwise op);

}  // namespace cuda
#endif
}  // namespace layers

Tensor::SPtr createElementwise(Tensor::SPtr, Tensor::SPtr, layers::Elementwise);

Tensor::SPtr add(Tensor::SPtr, Tensor::SPtr);
Tensor::SPtr operator+(Tensor::SPtr, Tensor::SPtr);

Tensor::SPtr sub(Tensor::SPtr, Tensor::SPtr);
Tensor::SPtr operator-(Tensor::SPtr, Tensor::SPtr);

Tensor::SPtr mul(Tensor::SPtr, Tensor::SPtr);
Tensor::SPtr operator*(Tensor::SPtr, Tensor::SPtr);

Tensor::SPtr div(Tensor::SPtr, Tensor::SPtr);
Tensor::SPtr operator/(Tensor::SPtr, Tensor::SPtr);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_ELEMENTWISE_H_
