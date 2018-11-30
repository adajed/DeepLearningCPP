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
    Memory output = mOutputs[0]->getMemory();

    if (output.getType() == MemoryType::kHOST_MEMORY)
    {
        std::memcpy(output.getValues(), input.data(),
                    output.getCount() * sizeof(float));
    }
#ifdef CUDA_AVAILABLE
    else  // output.getType() == MemoryType::kDEVICE_MEMORY
        cuda::copyInput(output.getValues(), input.data(), output.getCount());
#endif
}

}  // namespace core
}  // namespace graphdl
