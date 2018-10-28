#ifndef DLL_CORE_LAYERS_REDUCE_SUM_H_
#define DLL_CORE_LAYERS_REDUCE_SUM_H_

#include "gradientOper.h"

namespace dll
{
namespace core
{
namespace layers
{
class ReduceSumOper : public GradientOper
{
   public:
    ReduceSumOper(Tensor::SPtr tensor);

    GradientOper::TensorMap gradients(Tensor::SPtr out,
                                      Tensor::SPtr outGrad) override;

   private:
    void executeOper(const InputDict& inputs) override;
};

class ReduceSumGradientOper : public Oper
{
   public:
    ReduceSumGradientOper(Tensor::SPtr in, Tensor::SPtr out,
                          Tensor::SPtr outGrad);

   private:
    void executeOper(const InputDict& inputs) override;
};

}  // namespace layers

Tensor::SPtr reduceSum(Tensor::SPtr tensor);

}  // namespace core

}  // namespace dll

#endif  // DLL_CORE_LAYERS_REDUCE_SUM_H_
