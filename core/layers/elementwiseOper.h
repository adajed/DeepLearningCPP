#ifndef DLL_CORE_LAYERS_ELEMENTWISE_OPER_H_
#define DLL_CORE_LAYERS_ELEMENTWISE_OPER_H_

#include <functional>
#include "gradientOper.h"

namespace dll
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

class ElementwiseOper : public GradientOper
{
   public:
    ElementwiseOper(Tensor::SPtr t1, Tensor::SPtr t2, Elementwise op);

    TensorMap gradients(Tensor::SPtr output, Tensor::SPtr outputGrad) override;

   private:
    static std::vector<Tensor::SPtr> createOutputs(Tensor::SPtr t1,
                                                   Tensor::SPtr t2);

    void executeOper(const InputDict& inputs) override;

    Elementwise mOp;
    ElementwiseFun mFun;
};

class ElementwiseGradientOper : public Oper
{
   public:
    ElementwiseGradientOper(Tensor::SPtr t1, Tensor::SPtr t2, Tensor::SPtr out,
                            Tensor::SPtr outGrad, Elementwise op);

   private:
    static std::vector<Tensor::SPtr> createOutputs(Tensor::SPtr, Tensor::SPtr);

    void executeOper(const InputDict&) override;

    Elementwise mOp;
    ElementwiseFun mFun1, mFun2;
};

}  // namespace layers

Tensor::SPtr add(Tensor::SPtr, Tensor::SPtr);
Tensor::SPtr operator+(Tensor::SPtr, Tensor::SPtr);

Tensor::SPtr sub(Tensor::SPtr, Tensor::SPtr);
Tensor::SPtr operator-(Tensor::SPtr, Tensor::SPtr);

Tensor::SPtr mul(Tensor::SPtr, Tensor::SPtr);
Tensor::SPtr operator*(Tensor::SPtr, Tensor::SPtr);

Tensor::SPtr div(Tensor::SPtr, Tensor::SPtr);
Tensor::SPtr operator/(Tensor::SPtr, Tensor::SPtr);

}  // namespace core
}  // namespace dll

#endif  // DLL_CORE_LAYERS_ELEMENTWISE_OPER_H_
