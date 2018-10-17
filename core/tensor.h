#pragma once

#include "dll.h"

#include <vector>

namespace dll
{
namespace core
{

// forward declarations
class Tensor;

using TensorSPtr = SharedPtr<Tensor>;
using TensorVecUPtr = std::vector<TensorSPtr>;

//! \class Tensor
//! \brief Implementation of ITensor interface.
//!
class Tensor : public ITensor
{
public:
    std::string getName() const override;
    void setName(const std::string& name) override;

    Shape getShape() const override;
    void setShape(const Shape& shape) override;

    HostTensor eval(const InputDict& inputs) override;

private:
    std::string mName; //!< Tensor name.
    Shape mShape; //< Tensor shape.
};

} // namespace core
} // namespace dll
