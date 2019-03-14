#include "weights.h"

#include "graph.h"

#include <random>
#include <utility>

namespace graphdl
{
namespace core
{
WeightsLayer::WeightsLayer(ID id, const std::string& name, const Shape& shape,
                           Initializer::SPtr initializer, MemoryType type)
    : Layer(id, {}, {createTensor(name, shape, type)}),
      mInitializer(std::move(initializer))
{
}

void WeightsLayer::initialize()
{
    float* memory = mOutputs[0]->getMemory().getValues();
    MemoryType type = mOutputs[0]->getMemory().getType();
    mInitializer->init(memory, mOutputs[0]->getShape(), type);
}

// This does nothing, because weights are already in memory.
void WeightsLayer::execute(const std::vector<float*>& /*inputs*/,
                           const std::vector<float*>& /*outputs*/,
                           const InputDict& /*inputDict*/)
{
}

Tensor::SPtr weights(const std::string& name, const TensorShape& shape,
                     Initializer::SPtr initializer, MemoryType type,
                     const std::string& nspace)
{
    Layer::SPtr layer =
        createLayer<WeightsLayer>(name, shape, std::move(initializer), type);
    Tensor::SPtr tensor = getDefaultGraph()->addWeights(name, layer, nspace);
    return tensor;
}

}  // namespace core
}  // namespace graphdl
