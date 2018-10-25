#include "divOper.h"
#include "dll_errors.h"
#include "dll_ops.h"

namespace dll
{
namespace core
{
namespace
{
class DivGradientOper : public Oper
{
   public:
    DivGradientOper(Tensor::SPtr in1, Tensor::SPtr in2, Tensor::SPtr out)
        : Oper({in1, in2, out}, createOutputs(in1, in2, out))
    {
    }

   private:
    static std::vector<Tensor::SPtr> createOutputs(Tensor::SPtr in1,
                                                   Tensor::SPtr in2,
                                                   Tensor::SPtr out)
    {
        Tensor::SPtr grad1 = std::make_shared<Tensor>("", in1->shape());
        Tensor::SPtr grad2 = std::make_shared<Tensor>("", in2->shape());
        return {grad1, grad2};
    }

    void executeOper(const InputDict& inputs) override
    {
        Tensor::SPtr input1 = mInputs[0].lock();
        input1->exec(inputs);
        Tensor::SPtr input2 = mInputs[1].lock();
        input2->exec(inputs);

        Memory in1 = input1->getMemory();
        Memory in2 = input2->getMemory();

        Memory gradient1 = mOutputs[0]->getMemory();
        Memory gradient2 = mOutputs[1]->getMemory();

        for (std::size_t i = 0; i < in1.count(); ++i)
        {
            gradient1[i] = 1. / in2[i];
            gradient2[i] = -in1[i] / (in2[i] * in2[i]);
        }
    }
};

}  // namespace

std::map<Tensor::SPtr, GradientOper::TensorMap> DivOper::gradients()
{
    std::vector<Tensor::SPtr> inputs = getInputs();
    std::vector<Tensor::SPtr> outputs = getOutputs();

    Oper::SPtr gradOper =
        std::make_shared<DivGradientOper>(inputs[0], inputs[1], outputs[0]);
    getDefaultGraph()->insertOperation(gradOper);
    std::vector<Tensor::SPtr> grads = gradOper->getOutputs();

    return {{outputs[0], {{inputs[0], grads[0]}, {inputs[1], grads[1]}}}};
}

Tensor::SPtr div(Tensor::SPtr t1, Tensor::SPtr t2)
{
    if (t1->shape() != t2->shape())
        throw errors::NotMatchingShapesError();
    Oper::SPtr oper = std::make_shared<DivOper>(t1, t2);
    getDefaultGraph()->insertOperation(oper);
    return oper->getOutputs()[0];
}

Tensor::SPtr operator/(Tensor::SPtr t1, Tensor::SPtr t2) { return div(t1, t2); }

}  // namespace core

ITensorSPtr div(ITensorSPtr t1, ITensorSPtr t2)
{
    core::Tensor::SPtr tensor1 = std::static_pointer_cast<core::Tensor>(t1);
    core::Tensor::SPtr tensor2 = std::static_pointer_cast<core::Tensor>(t2);
    return ITensorSPtr(core::div(tensor1, tensor2));
}

ITensorSPtr operator/(ITensorSPtr t1, ITensorSPtr t2) { return div(t1, t2); }

}  // namespace dll
