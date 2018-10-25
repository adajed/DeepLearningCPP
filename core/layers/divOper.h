#ifndef DLL_CORE_DIV_OPER_H_
#define DLL_CORE_DIV_OPER_H_

#include "dll_ops.h"
#include "elementwiseOper.h"
#include "graph.h"

namespace dll
{
namespace core
{
class DivOper : public ElementwiseOper
{
   public:
    DivOper(Tensor::SPtr t1, Tensor::SPtr t2) : ElementwiseOper(t1, t2) {}

    std::map<Tensor::SPtr, GradientOper::TensorMap> gradients() override;

   private:
    float elementwise(float f1, float f2) override { return f1 / f2; }
};

Tensor::SPtr div(Tensor::SPtr, Tensor::SPtr);
Tensor::SPtr operator/(Tensor::SPtr, Tensor::SPtr);

}  // namespace core
}  // namespace dll

#endif  // DLL_CORE_DIV_OPER_H_
