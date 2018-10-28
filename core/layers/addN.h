#ifndef DLL_CORE_LAYERS_ADD_N_H_
#define DLL_CORE_LAYERS_ADD_N_H_

#include "gradientOper.h"

namespace dll
{
namespace core
{
namespace layers
{
class AddNOper : public GradientOper
{
   public:
    AddNOper(std::vector<Tensor::SPtr> tensors);

    GradientOper::TensorMap gradients(Tensor::SPtr out,
                                      Tensor::SPtr outGrad) override;

   private:
    void executeOper(const InputDict& inputs) override;
};

class AddNGradientOper : public Oper
{
   public:
    AddNGradientOper(std::vector<Tensor::SPtr> ins, Tensor::SPtr out,
                     Tensor::SPtr outGrad);

   private:
    void executeOper(const InputDict& inputs) override;
};

}  // namespace layers

Tensor::SPtr addN(std::vector<Tensor::SPtr> tensors);

}  // namespace core
}  // namespace dll

#endif  // DLL_CORE_LAYERS_ADD_N_H_
