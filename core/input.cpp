#include "input.h"

#include <cstring>

namespace graphdl
{
namespace core
{
InputLayer::InputLayer(ID id, const std::string& name, const Shape& shape,
                       MemoryType type)
    : Layer(id, {}, {createTensor(name, shape, type)})
{
}

std::set<Tensor::SPtr> InputLayer::getNecessaryInputs() const
{
    return {mOutputs[0]};
}

void InputLayer::execute(const InputDict& inputs)
{
    std::string name = mOutputs[0]->getName();
    HostTensor input = inputs.at(name);
    Memory<float> output = mOutputs[0]->getMemory();

    output.fillFrom(input.data());
}

}  // namespace core
}  // namespace graphdl
