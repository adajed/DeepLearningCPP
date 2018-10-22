#ifndef DLL_CORE_INPUT_H_
#define DLL_CORE_INPUT_H_

#include <assert.h>
#include "oper.h"

namespace dll
{
namespace core
{

class InputOper : public Oper
{
public:
    InputOper(const std::string& name, const Shape& shape)
        : Oper({}, createOutputs(name, shape))
    {}

private:
    static std::vector<Tensor::SPtr> createOutputs(const std::string& name, const Shape& shape)
    {
        return {std::make_shared<Tensor>(name, shape)};
    }

    void executeOper(const InputDict& inputs) override
    {
        std::string name = mOutputs[0]->getName();
        HostTensor input = inputs.at(name);
        Memory output = mOutputs[0]->getMemory();

        assert(output.isAllocated());
        for (std::size_t i = 0; i < input.count; ++i)
            output.getValues()[i] = input.values[i];
    }
};

} // namespace core
} // namespace dll

#endif // DLL_CORE_INPUT_H_
