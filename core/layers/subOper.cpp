#include "subOper.h"
#include "dll_errors.h"
#include "dll_ops.h"

namespace dll
{
namespace core
{
namespace
{
class SubGradientOper : public Oper
{
   public:
    SubGradientOper(Tensor::SPtr in1, Tensor::SPtr in2, Tensor::SPtr out, Tensor::SPtr outGrad)
        : Oper({in1, in2, out, outGrad}, createOutputs(in1, in2))
    {
    }

   private:
    static std::vector<Tensor::SPtr> createOutputs(Tensor::SPtr in1,
                                                   Tensor::SPtr in2)
    {
        Tensor::SPtr grad1 = std::make_shared<Tensor>("", in1->shape());
        Tensor::SPtr grad2 = std::make_shared<Tensor>("", in2->shape());
        return {grad1, grad2};
    }

    void executeOper(const InputDict& inputs) override
    {
        Tensor::SPtr outputGrad = mInputs[3].lock();
        outputGrad->exec(inputs);

        Memory outGrad = outputGrad->getMemory();
        Memory in1Grad = mOutputs[0]->getMemory();
        Memory in2Grad = mOutputs[1]->getMemory();

        for (std::size_t i = 0; i < outGrad.count(); ++i)
        {
            in1Grad[i] = outGrad[i];
            in2Grad[i] = -outGrad[i];
        }
    }
};

}  // namespace

GradientOper::TensorMap SubOper::gradients(Tensor::SPtr output, Tensor::SPtr outputGrad)
{
    assert(output == mOutputs[0]);

    std::vector<Tensor::SPtr> inputs = getInputs();

    Oper::SPtr gradOper =
        std::make_shared<SubGradientOper>(inputs[0], inputs[1], output, outputGrad);
    getDefaultGraph()->insertOperation(gradOper);
    std::vector<Tensor::SPtr> grads = gradOper->getOutputs();

    return {{inputs[0], grads[0]}, {inputs[1], grads[1]}};
}

Tensor::SPtr sub(Tensor::SPtr t1, Tensor::SPtr t2)
{
    if (t1->shape() != t2->shape())
        throw errors::NotMatchingShapesError();
    Oper::SPtr oper = std::make_shared<SubOper>(t1, t2);
    getDefaultGraph()->insertOperation(oper);
    return oper->getOutputs()[0];
}

Tensor::SPtr operator-(Tensor::SPtr t1, Tensor::SPtr t2) { return sub(t1, t2); }

}  // namespace core

ITensorSPtr sub(ITensorSPtr t1, ITensorSPtr t2)
{
    core::Tensor::SPtr tensor1 = std::static_pointer_cast<core::Tensor>(t1);
    core::Tensor::SPtr tensor2 = std::static_pointer_cast<core::Tensor>(t2);
    return ITensorSPtr(core::sub(tensor1, tensor2));
}

ITensorSPtr operator-(ITensorSPtr t1, ITensorSPtr t2) { return sub(t1, t2); }

}  // namespace dll
