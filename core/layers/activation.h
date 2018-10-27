#ifndef DLL_CORE_LAYERS_ACTIVATION_OPER_H_
#define DLL_CORE_LAYERS_ACTIVATION_OPER_H_

#include <functional>
#include "gradientOper.h"

namespace dll
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
    kSQRT = 4,
    kABS = 5,
    kNEG = 6,
    kRECIPROCAL = 7
};

class ActivationOper : public GradientOper
{
   public:
    ActivationOper(Tensor::SPtr, Activation);

    GradientOper::TensorMap gradients(Tensor::SPtr, Tensor::SPtr) override;

   private:
    void executeOper(const InputDict&) override;

    Activation mOp;
    std::function<float(float)> mFun;
};

class ActivationGradientOper : public Oper
{
   public:
    ActivationGradientOper(Tensor::SPtr, Tensor::SPtr, Tensor::SPtr,
                           Activation);

   private:
    void executeOper(const InputDict&) override;

    Activation mOp;
    std::function<float(float, float)> mFun;
};

Tensor::SPtr createActivation(Tensor::SPtr, Activation);

}  // namespace layers

Tensor::SPtr relu(Tensor::SPtr);
Tensor::SPtr sigmoid(Tensor::SPtr);
Tensor::SPtr tanh(Tensor::SPtr);
Tensor::SPtr square(Tensor::SPtr);
Tensor::SPtr sqrt(Tensor::SPtr);
Tensor::SPtr abs(Tensor::SPtr);
Tensor::SPtr neg(Tensor::SPtr);
Tensor::SPtr reciprocal(Tensor::SPtr);

}  // namespace core
}  // namespace dll

#endif  // DLL_CORE_LAYERS_ACTIVATION_OPER_H_
