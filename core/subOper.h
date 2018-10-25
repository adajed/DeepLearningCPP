#ifndef DLL_CORE_SUB_OPER_H_
#define DLL_CORE_SUB_OPER_H_

#include "dll_ops.h"
#include "graph.h"
#include "elementwiseOper.h"

namespace dll
{
namespace core
{

class SubOper : public ElementwiseOper
{
public:
    SubOper(Tensor::SPtr t1, Tensor::SPtr t2)
        : ElementwiseOper(t1, t2)
    {}

    std::map<Tensor::SPtr, GradientOper::TensorMap> gradients() override;

private:
    float elementwise(float f1, float f2) override
    {
        return f1 - f2;
    }
};

Tensor::SPtr sub(Tensor::SPtr t1, Tensor::SPtr t2);
Tensor::SPtr operator -(Tensor::SPtr t1, Tensor::SPtr t2);

} // namespace core
} // namespace dll

#endif // DLL_CORE_SUB_OPER_H_
