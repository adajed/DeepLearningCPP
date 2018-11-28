#include "abstractInitializer.h"

#include <utility>

namespace graphdl
{
namespace core
{
AbstractInitializer::AbstractInitializer(Initializer::SPtr initializer)
    : mInitializer(std::move(std::move(initializer)))
{
}

void AbstractInitializer::init(float* memory, const Shape& shape,
                               MemoryLocation location) const
{
    mInitializer->init(memory, shape, core::memoryLocationToType(location));
}

Initializer::SPtr AbstractInitializer::get()
{
    return mInitializer;
}

SharedPtr<AbstractInitializer> castIInitializer(
    const SharedPtr<IInitializer>& initializer)
{
    return std::static_pointer_cast<AbstractInitializer>(initializer);
}

SharedPtr<AbstractInitializer> makeAbstractInitializer(
    Initializer::SPtr initializer)
{
    return std::make_shared<AbstractInitializer>(initializer);
}

}  // namespace core
}  // namespace graphdl
