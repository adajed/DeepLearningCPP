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

private:
    float elementwise(float f1, float f2)
    {
        return f1 - f2;
    }
};

} // namespace core
} // namespace dll

#endif // DLL_CORE_SUB_OPER_H_
