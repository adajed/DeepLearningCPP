#ifndef DLL_CORE_DIV_OPER_H_
#define DLL_CORE_DIV_OPER_H_

#include "dll_ops.h"
#include "elementwiseOper.h"
#include "graph.h"

namespace dll
{
namespace core
{
namespace layers
{
class DivOper : public ElementwiseOper
{
   public:
    DivOper(Tensor::SPtr t1, Tensor::SPtr t2) : ElementwiseOper(t1, t2) {}

    TensorMap gradients(Tensor::SPtr output, Tensor::SPtr outputGrad) override;

   private:
    float elementwise(float f1, float f2) override { return f1 / f2; }
};

class DivGradientOper : public Oper
{
   public:
    DivGradientOper(Tensor::SPtr in1, Tensor::SPtr in2, Tensor::SPtr out,
                    Tensor::SPtr outGrad);

   private:
    static std::vector<Tensor::SPtr> createOutputs(Tensor::SPtr in1,
                                                   Tensor::SPtr in2);

    void executeOper(const InputDict& inputs) override;
};

}  // namespace layers

Tensor::SPtr div(Tensor::SPtr, Tensor::SPtr);
Tensor::SPtr operator/(Tensor::SPtr, Tensor::SPtr);

}  // namespace core
}  // namespace dll

#endif  // DLL_CORE_DIV_OPER_H_
