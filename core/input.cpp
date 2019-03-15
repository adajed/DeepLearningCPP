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

void InputLayer::execute(const std::vector<float*>& /*inputs*/,
                         const std::vector<float*>& outputs,
                         const InputDict& inputDict)
{
    std::string name = mOutputs[0]->getName();
    HostTensor host = inputDict.at(name);
    size_t size = mOutputs[0]->getCount();

    if (mOutputs[0]->getType() == MemoryType::kHOST_MEMORY)
    {
        std::memcpy(outputs[0], host.data(), size * sizeof(float));
    }
#ifdef CUDA_AVAILABLE
    else  // output.getType() == MemoryType::kDEVICE_MEMORY
        cuda::copyInput(outputs[0], host.data(), size);
#endif
}

}  // namespace core
}  // namespace graphdl
