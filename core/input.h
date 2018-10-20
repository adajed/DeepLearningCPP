#ifndef DLL_CORE_INPUT_H_
#define DLL_CORE_INPUT_H_

#include "oper.h"

namespace dll
{
namespace core
{

class InputOper : public Oper
{
public:
    InputOper(const std::string& name, const Shape& shape)
        : Oper({}, {createTensor(this, name, shape)})
    {}

private:
    void executeOper(const InputDict& inputs) override
    {
        std::string name = mOutputs[0]->getName();
        HostTensor input = inputs.at(name);
        mOutputs[0]->setHostTensor(input);
    }
};

} // namespace core
} // namespace dll

#endif // DLL_CORE_INPUT_H_
