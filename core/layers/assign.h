#ifndef DLL_CORE_LAYERS_ASSIGN_OPER_H_
#define DLL_CORE_LAYERS_ASSIGN_OPER_H_

#include "oper.h"

namespace dll
{
namespace core
{
namespace layers
{
class AssignOper : public Oper
{
   public:
    AssignOper(Tensor::SPtr dest, Tensor::SPtr src);

   private:
    void executeOper(const InputDict& inputs) override;

    Tensor::WeakPtr mDest;
};

}  // namespace layers

Tensor::SPtr assign(Tensor::SPtr dest, Tensor::SPtr src);

}  // namespace core
}  // namespace dll

#endif  // DLL_CORE_LAYERS_ASSIGN_OPER_H_
