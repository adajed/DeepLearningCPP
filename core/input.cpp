#include "input.h"

namespace graphdl
{
namespace core
{
InputLayer::InputLayer(Graph::SPtr graph,
                       const std::string& name,
                       const Shape& shape)
    : Oper(graph, {} {createTensor(name, shape)})
{
}

void InputLayer::execute(const InputDict& inputs)
{
    std::string name = mOutputs[0]->getName();
    HostTensor input = inputs.at(name);
    Memory output = mOutputs[0]->getMemory();

    for (std::size_t i = 0; i < input.count; ++i)
        output.getValues()[i] = input.values[i];
}

}  // namespace core
}  // namespace graphdl
