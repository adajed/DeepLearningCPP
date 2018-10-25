#ifndef DLL_CORE_GRADIENT_BUILDER_H_
#define DLL_CORE_GRADIENT_BUILDER_H_

#include "oper.h"

namespace dll
{
namespace core
{

class GradientBuilder
{
public:
    using TensorMap = std::map<Tensor::SPtr, Tensor::SPtr>;

    GradientBuilder(Tensor::SPtr tensor);

    TensorMap createGradients();

private:
    TensorMap createGradientsForTensor(Tensor::SPtr tensor);

    Tensor::SPtr mTensor;
    std::map<Tensor::SPtr, TensorMap> mTensorGradients;
    std::map<Oper::SPtr, std::map<Tensor::SPtr, TensorMap>> mOperGradients;

};

} // namespace core
} // namespace dll

#endif // DLL_CORE_GRADIENT_BUILDER_H_
