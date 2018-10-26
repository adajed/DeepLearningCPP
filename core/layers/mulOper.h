#ifndef DLL_CORE_MUL_OPER_H_
#define DLL_CORE_MUL_OPER_H_

#include "elementwiseOper.h"

namespace dll
{
namespace core
{
namespace layers
{
class MulOper : public ElementwiseOper
{
   public:
    MulOper(Tensor::SPtr t1, Tensor::SPtr t2) : ElementwiseOper(t1, t2) {}

    TensorMap gradients(Tensor::SPtr output, Tensor::SPtr outputGrad) override;

   private:
    float elementwise(float f1, float f2) override { return f1 * f2; }
};

class MulGradientOper : public Oper
{
   public:
    MulGradientOper(Tensor::SPtr t1, Tensor::SPtr t2, Tensor::SPtr out,
                    Tensor::SPtr outGrad);

   private:
    static std::vector<Tensor::SPtr> createOutputs(Tensor::SPtr t1,
                                                   Tensor::SPtr t2);

    void executeOper(const InputDict& inputs) override;
};

}  // namespace layers

Tensor::SPtr mul(Tensor::SPtr, Tensor::SPtr);
Tensor::SPtr operator*(Tensor::SPtr, Tensor::SPtr);

}  // namespace core
}  // namespace dll

#endif  // DLL_CORE_MUL_OPER_H_
