#pragma once

#include "dll.h"

#include <vector>

namespace dll
{
namespace core
{

// forward declarations
class Tensor;

using TensorUPtr = SharedPtr<Tensor>;
using TensorVecUPtr = std::vector<TensorUPtr>;

//! \class Tensor
//! \brief Implementation of ITensor interface.
//!
class Tensor : public ITensor
{
public:
    std::string getName() const override;
    void setName(std::string const& name) override;

    Shape getShape() const override;
    void setShape(const Shape& shape) override;

    HostTensor eval(InputDict const& inputs) override;

private:
    std::string mName; //!< Tensor name.
    Shape mShape; //< Tensor shape.
};

} // namespace core
} // namespace dll
