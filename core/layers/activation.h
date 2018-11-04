#ifndef GRAPHDL_CORE_LAYERS_ACTIVATION_H_
#define GRAPHDL_CORE_LAYERS_ACTIVATION_H_

#include <functional>
#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
enum class Activation
{
    kRELU = 0,
    kSIGMOID = 1,
    kTANH = 2,
    kSQUARE = 3,
    kABS = 4,
    kNEG = 5,
    kRECIPROCAL = 6,
    kLOG = 7
};

class ActivationLayer : public DifferentiableLayer
{
   public:
    ActivationLayer(ID, Tensor::SPtr, Activation);

    TensorMap gradients(Tensor::SPtr, Tensor::SPtr) override;

   private:
    void execute(const InputDict&) override;

    Activation mOp;
    std::function<float(float)> mFun;
};

class ActivationGradientLayer : public Layer
{
   public:
    ActivationGradientLayer(ID, Tensor::SPtr, Tensor::SPtr, Tensor::SPtr,
                            Activation);

   private:
    void execute(const InputDict&) override;

    Activation mOp;
    std::function<float(float, float)> mFun;
};

namespace cuda
{
extern "C" void runActivationDevice(std::size_t size, float* x, float* y,
                                    Activation op);

extern "C" void runActivationGradientDevice(std::size_t size, float* x,
                                            float* y, float* yGrad,
                                            float* xGrad, Activation op);
}  // namespace cuda

}  // namespace layers

Tensor::SPtr createActivation(Tensor::SPtr, layers::Activation);

Tensor::SPtr relu(Tensor::SPtr);
Tensor::SPtr sigmoid(Tensor::SPtr);
Tensor::SPtr tanh(Tensor::SPtr);
Tensor::SPtr square(Tensor::SPtr);
Tensor::SPtr abs(Tensor::SPtr);
Tensor::SPtr neg(Tensor::SPtr);
Tensor::SPtr reciprocal(Tensor::SPtr);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_ACTIVATION_H_
