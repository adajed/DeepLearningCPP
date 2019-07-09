#ifndef GRAPHDL_CORE_LAYERS_ACTIVATION_H_
#define GRAPHDL_CORE_LAYERS_ACTIVATION_H_

#include "differentiableLayer.h"

#include <functional>

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
    kLOG = 7,
    kSQRT = 8,
    kEXP = 9,
    kLEAKY_RELU = 10,
    kRELU_6 = 11,
    kELU = 12,
    kSOFTPLUS = 13,
    kSOFTSIGN = 14
};

class ActivationLayer : public DifferentiableLayer
{
  public:
    ActivationLayer(ID, const Tensor::SPtr&, Activation);

    TensorMap gradients(Tensor::SPtr, Tensor::SPtr) override;

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    Activation mOp;
    std::function<float(float)> mFun;
};

class ActivationGradientLayer : public Layer
{
  public:
    ActivationGradientLayer(ID, const Tensor::SPtr&, const Tensor::SPtr&,
                            const Tensor::SPtr&, Activation);

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    Activation mOp;
    std::function<float(float, float)> mFun;
};

#ifdef CUDA_AVAILABLE
namespace cuda
{
void runActivationDevice(const float* x, float* y, size_t size, Activation op);

void runActivationGradientDevice(const float* x, const float* y,
                                 const float* yGrad, float* xGrad, size_t size,
                                 Activation op);
}  // namespace cuda
#endif

void runActivationHost(const float* x, float* y, size_t size, Activation op);

void runActivationGradientHost(const float* x, const float* y,
                               const float* yGrad, float* xGrad, size_t size,
                               Activation op);

}  // namespace layers

Tensor::SPtr createActivation(Tensor::SPtr, layers::Activation);

Tensor::SPtr relu(Tensor::SPtr);
Tensor::SPtr sigmoid(Tensor::SPtr);
Tensor::SPtr tanh(Tensor::SPtr);
Tensor::SPtr square(Tensor::SPtr);
Tensor::SPtr abs(Tensor::SPtr);
Tensor::SPtr neg(Tensor::SPtr);
Tensor::SPtr reciprocal(Tensor::SPtr);
Tensor::SPtr log(Tensor::SPtr);
Tensor::SPtr sqrt(Tensor::SPtr);
Tensor::SPtr exp(Tensor::SPtr);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_ACTIVATION_H_
