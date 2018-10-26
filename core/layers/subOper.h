#ifndef DLL_CORE_SUB_OPER_H_
#define DLL_CORE_SUB_OPER_H_

#include "dll_ops.h"
#include "elementwiseOper.h"
#include "graph.h"

namespace dll
{
namespace core
{
namespace layers
{
class SubOper : public ElementwiseOper
{
   public:
    SubOper(Tensor::SPtr t1, Tensor::SPtr t2) : ElementwiseOper(t1, t2) {}

    TensorMap gradients(Tensor::SPtr output, Tensor::SPtr outputGrad) override;

   private:
    float elementwise(float f1, float f2) override { return f1 - f2; }
};

class SubGradientOper : public Oper
{
   public:
    SubGradientOper(Tensor::SPtr t1, Tensor::SPtr t2, Tensor::SPtr out,
                    Tensor::SPtr outGrad);

   private:
    static std::vector<Tensor::SPtr> createOutputs(Tensor::SPtr t1,
                                                   Tensor::SPtr t2);

    void executeOper(const InputDict& inputs) override;
};

}  // namespace layers

Tensor::SPtr sub(Tensor::SPtr t1, Tensor::SPtr t2);
Tensor::SPtr operator-(Tensor::SPtr t1, Tensor::SPtr t2);

}  // namespace core
}  // namespace dll

#endif  // DLL_CORE_SUB_OPER_H_
