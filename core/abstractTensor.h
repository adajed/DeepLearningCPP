#ifndef GRAPHDL_CORE_ABSTRACT_TENSOR_H_
#define GRAPHDL_CORE_ABSTRACT_TENSOR_H_

#include "graphdl.h"
#include "layer.h"

#include <memory>

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
    using Ptr = std::shared_ptr<AbstractTensor>;

    AbstractTensor(Tensor::SPtr tensor);

    std::string getName() const override;

    void setName(const std::string& name) override;

    Shape getShape() const override;

    HostTensor eval(const InputDict& inputs) override;

    Tensor::SPtr get() const;

  private:
    Tensor::SPtr mTensor;
};

AbstractTensor::Ptr makeAbstractTensor(Tensor::SPtr tensor);

AbstractTensor::Ptr castITensorPtr(ITensorPtr tensor);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_ABSTRACT_TENSOR_H_
