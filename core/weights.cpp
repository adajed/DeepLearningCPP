#include "weights.h"

#include <random>

namespace graphdl
{
namespace core
{
WeightsLayer::WeightsLayer(ID id, const std::string& name, const Shape& shape)
    : Layer(id, {}, {createTensor(name, shape)})
{
}

void WeightsLayer::initialize()
{
    Memory memory = mOutputs[0]->getMemory();

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(-1., 1.);

    for (std::size_t i = 0; i < memory.getCount(); ++i) memory[i] = dist(e2);
}

// This does nothing, because weights are already in memory.
void WeightsLayer::execute(const InputDict& inputs) {}

}  // namespace core
}  // namespace graphdl
