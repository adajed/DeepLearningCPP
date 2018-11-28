#include "uniformInitializer.h"

#ifdef CUDA_AVAILABLE
#include "layers/cuda/randomUtils.h"
#endif
#include "abstractInitializer.h"

#include <random>

namespace graphdl
{
namespace core
{
namespace initializers
{
UniformInitializer::UniformInitializer(float min, float max, size_t seed)
    : mMinValue(min), mMaxValue(max), mSeed(seed)
{
}

void UniformInitializer::initHost(float* memory, const TensorShape& shape) const
{
    std::uniform_real_distribution<> d(mMinValue, mMaxValue);
    std::mt19937 e2(mSeed);

    for (size_t pos = 0; pos < shape.getCount(); ++pos) memory[pos] = d(e2);
}

void UniformInitializer::initDevice(float* memory,
                                    const TensorShape& shape) const
{
#ifdef CUDA_AVAILABLE
    cuda::uniformRandom(memory, shape.getCount(), mMinValue, mMaxValue, mSeed);
#else
    throw std::runtime_error("Cuda not available, please use CPU");
#endif
}

}  // namespace initializers
}  // namespace core

SharedPtr<IInitializer> uniformInitializer(float min, float max, size_t seed)
{
    auto init = std::make_shared<core::initializers::UniformInitializer>(
        min, max, seed);
    return core::makeAbstractInitializer(init);
}

}  // namespace graphdl
