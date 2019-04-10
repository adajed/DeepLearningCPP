#include "constantInitializer.h"

#include "abstractInitializer.h"
#include "graphdl.h"
#ifdef CUDA_AVAILABLE
#include "layers/cuda/utils.h"
#endif

#include <algorithm>

namespace graphdl
{
namespace core
{
namespace initializers
{
ConstantInitializer::ConstantInitializer(float value)
    : Initializer(0), mValue(value)
{
}

void ConstantInitializer::initHost(float* memory, const TensorShape& shape)
{
    std::fill_n(memory, shape.getCount(), mValue);
}

// NOLINTNEXTLINE
void ConstantInitializer::initDevice(float* memory, const TensorShape& shape)
{
#ifdef CUDA_AVAILABLE
    layers::cuda::utils::fill(memory, shape.getCount(), mValue);
#else
    throw std::runtime_error("Cuda not available, please use CPU");
#endif
}

}  // namespace initializers

initializers::Initializer::SPtr constantInitializer(float value)
{
    return std::make_shared<initializers::ConstantInitializer>(value);
}

}  // namespace core

SharedPtr<IInitializer> constantInitializer(float value)
{
    auto init =
        std::make_shared<core::initializers::ConstantInitializer>(value);
    return core::makeAbstractInitializer(init);
}

}  // namespace graphdl
