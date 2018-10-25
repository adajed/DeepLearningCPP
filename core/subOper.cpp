#include "dll_ops.h"
#include "subOper.h"

namespace dll
{
namespace core
{
namespace
{
class SubGradientOper : public Oper
{
public:
    SubGradientOper(Tensor::SPtr in1, Tensor::SPtr in2, Tensor::SPtr out)
        : Oper({in1, in2, out}, createOutputs(in1, in2, out))
    {
    }

    void initialize() override
    {
        Memory gradient1 = mOutputs[0]->getMemory();
        Memory gradient2 = mOutputs[1]->getMemory();

        for (std::size_t i = 0; i < gradient1.count(); ++i)
            gradient1[i] = 1.;
        for (std::size_t i = 0; i < gradient2.count(); ++i)
            gradient2[i] = -1.;
    }

private:
    static std::vector<Tensor::SPtr> createOutputs(
            Tensor::SPtr in1, Tensor::SPtr in2, Tensor::SPtr out)
    {
        Tensor::SPtr grad1 = std::make_shared<Tensor>("", in1->shape());
        Tensor::SPtr grad2 = std::make_shared<Tensor>("", in2->shape());
        return {grad1, grad2};
    }

    //! Gradients are already calucated in initialize
    void executeOper(const InputDict& inputs) override
    {
    }

};

}  // namespace anonymous

std::map<Tensor::SPtr, GradientOper::TensorMap> SubOper::gradients()
{
    std::vector<Tensor::SPtr> inputs = getInputs();
    std::vector<Tensor::SPtr> outputs = getOutputs();

    Oper::SPtr gradOper = std::make_shared<SubGradientOper>(
            inputs[0], inputs[1], outputs[0]);
    getDefaultGraph()->insertOperation(gradOper);
    std::vector<Tensor::SPtr> grads = gradOper->getOutputs();

    return {{outputs[0], {{inputs[0], grads[0]}, {inputs[1], grads[1]}}}};
}

Tensor::SPtr sub(Tensor::SPtr t1, Tensor::SPtr t2)
{
    Oper::SPtr oper = std::make_shared<SubOper>(t1, t2);
    getDefaultGraph()->insertOperation(oper);
    return oper->getOutputs()[0];
}

Tensor::SPtr operator -(Tensor::SPtr t1, Tensor::SPtr t2)
{
    return sub(t1, t2);
}

}  // namespace core

ITensorSPtr sub(ITensorSPtr t1, ITensorSPtr t2)
{
    core::Tensor::SPtr tensor1 = std::static_pointer_cast<core::Tensor>(t1);
    core::Tensor::SPtr tensor2 = std::static_pointer_cast<core::Tensor>(t2);
    return ITensorSPtr(core::sub(tensor1, tensor2));
}

ITensorSPtr operator -(ITensorSPtr t1, ITensorSPtr t2)
{
    return sub(t1, t2);
}

}  // namespace dll
