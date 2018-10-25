#ifndef DLL_CORE_WEIGHTS_H_
#define DLL_CORE_WEIGHTS_H_

#include <random>

#include "oper.h"

namespace dll
{
namespace core
{
class WeightsOper : public Oper
{
   public:
    WeightsOper(const std::string& name, const Shape& shape)
        : Oper({}, createOutputs(name, shape))
    {
    }

    void initialize() override
    {
        Memory memory = mOutputs[0]->getMemory();

        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(-1., 1.);

        for (std::size_t i = 0; i < memory.count(); ++i)
            memory[i] = dist(e2);
    }

   private:
    static std::vector<Tensor::SPtr> createOutputs(const std::string& name,
                                                   const Shape& shape)
    {
        return {std::make_shared<Tensor>(name, shape)};
    }

    //! This method does nothing, because weights are already
    //!     in the tensor.
    void executeOper(const InputDict& inputs) override {}

};

}  // namespace core
}  // namespace dll

#endif  // DLL_CORE_WEIGHTS_H_
