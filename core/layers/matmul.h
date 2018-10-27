#ifndef DLL_CORE_LAYERS_MATMUL_H_
#define DLL_CORE_LAYERS_MATMUL_H_

#include "gradientOper.h"

namespace dll
{
namespace core
{
namespace layers
{
class MatmulOper : public GradientOper
{
public:
    MatmulOper(Tensor::SPtr m1, Tensor::SPtr m2);

    GradientOper::TensorMap gradients(Tensor::SPtr output,
                                      Tensor::SPtr outputGrad) override;
private:
    void executeOper(const InputDict& inputs) override;

};

class MatmulGradientOper : public Oper
{
   public:
    MatmulGradientOper(Tensor::SPtr m1, Tensor::SPtr m2, Tensor::SPtr out, Tensor::SPtr outGrad);

   private:
    void executeOper(const InputDict&) override;

};

}  // namespace layers

Tensor::SPtr matmul(Tensor::SPtr, Tensor::SPtr);

}  // namespace core
}  // namespace dll

#endif  // DLL_CORE_LAYERS_MATMUL_H_
