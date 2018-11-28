#include "constantInitializer.h"

#include "abstractInitializer.h"
#include "graphdl.h"

#include <algorithm>

namespace graphdl
{
namespace core
{
namespace initializers
{
ConstantInitializer::ConstantInitializer(float value) : mValue(value) {}

void ConstantInitializer::initHost(float* memory,
                                   const TensorShape& shape) const
{
    std::fill_n(memory, shape.getCount(), mValue);
}

void ConstantInitializer::initDevice(float* memory,
                                     const TensorShape& shape) const
{
#ifdef CUDA_AVAILABLE
    /* cuda::utils::fill(memory, mValue); */
#else
    throw std::runtime_error("Cuda not available, please use CPU");
#endif
}

}  // namespace initializers
}  // namespace core

SharedPtr<IInitializer> constantInitializer(float value)
{
    auto init =
        std::make_shared<core::initializers::ConstantInitializer>(value);
    return core::makeAbstractInitializer(init);
}

}  // namespace graphdl
