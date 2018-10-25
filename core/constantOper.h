#ifndef DLL_CORE_CONSTANT_OPER_H_
#define DLL_CORE_CONSTANT_OPER_H_

#include "oper.h"

namespace dll
{
namespace core
{
class ConstantOper : public Oper
{
public:
    ConstantOper(float value, const Shape& shape)
        : Oper({}, createOutputs(shape)), mValue(value)
    {
    }

    void initialize() override
    {
        Memory out = mOutputs[0]->getMemory();
        for (std::size_t i = 0; i < out.count(); ++i)
            out[i] = mValue;
    }

private:
    static std::vector<Tensor::SPtr> createOutputs(const Shape& shape)
    {
        return {std::make_shared<Tensor>("", shape)};
    }

    //! This method does nothing, because tensor is already
    //!     filled during initialize.
    void executeOper(const InputDict& inputs) override
    {
    }

    float mValue;

};

Tensor::SPtr constant(float value, const Shape& shape);

}  // namespace core
}  // namespace dll

#endif  // DLL_CORE_CONSTANT_OPER_H_
