#include "normalInitializer.h"

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
NormalInitializer::NormalInitializer(float mean, float stddev, size_t seed)
    : Initializer(seed), mMean(mean), mStddev(stddev)
{
}

void NormalInitializer::initHost(float* memory, const TensorShape& shape)
{
    std::normal_distribution<> d(mMean, mStddev);
    std::mt19937 gen(mSeed);

    for (size_t pos = 0; pos < shape.getCount(); ++pos) memory[pos] = d(gen);
}

void NormalInitializer::initDevice(float* memory, const TensorShape& shape)
{
#ifdef CUDA_AVAILABLE
    cuda::normalRandom(memory, shape.getCount(), mMean, mStddev, mSeed);
#else
    throw std::runtime_error("Cuda not available, please use CPU");
#endif
}

}  // namespace initializers
}  // namespace core

SharedPtr<IInitializer> normalInitializer(float mean, float stddev, size_t seed)
{
    auto init = std::make_shared<core::initializers::NormalInitializer>(
        mean, stddev, seed);
    return core::makeAbstractInitializer(init);
}

}  // namespace graphdl
