#ifndef DLL_CORE_GRADIENT_OPER_H_
#define DLL_CORE_GRADIENT_OPER_H_

#include "oper.h"

namespace dll
{
namespace core
{
class GradientOper : public Oper
{
   public:
    using TensorMap = std::map<Tensor::SPtr, Tensor::SPtr>;

    GradientOper(const std::vector<Tensor::SPtr>& inputs,
                 std::vector<Tensor::SPtr> outputs)
        : Oper(inputs, outputs)
    {
    }

    //! \fn gradients
    virtual TensorMap gradients(Tensor::SPtr output,
                                Tensor::SPtr outputGrad) = 0;

    bool hasGradient() const override final { return true; }
};

}  // namespace core
}  // namespace dll

#endif  // DLL_CORE_GRADIENT_OPER_H_
