#ifndef GRAPHDL_CORE_ABSTRACT_TENSOR_H_
#define GRAPHDL_CORE_ABSTRACT_TENSOR_H_

#include <memory>
#include "graphdl.h"
#include "oper.h"

namespace graphdl
{
namespace core
{
//! \class AbstractTensor
//! \brief Implementation of ITensor.
//!
class AbstractTensor : public ITensor
{
   public:
    using Ptr = std::unique_ptr<AbstractTensor>;

    AbstractTensor(Tensor::SPtr tensor);

    std::string getName() const override;

    void setName(const std::string& name) override;

    Shape getShape() const override;

    void eval(const InputDict& inputs, HostTensor hostTensor) override;

   private:
    Tensor::SPtr mTensor;
};

AbstractTensor::Ptr makeAbstract(Tensor::SPtr tensor);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_ABSTRACT_TENSOR_H_
