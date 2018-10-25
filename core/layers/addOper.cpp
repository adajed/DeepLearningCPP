#include "addOper.h"
#include "dll_ops.h"
#include "graph.h"

namespace dll
{
namespace core
{
namespace
{
class AddGradientOper : public Oper
{
   public:
    AddGradientOper(Tensor::SPtr input1, Tensor::SPtr input2,
                    Tensor::SPtr output)
        : Oper({input1, input2, output}, createOutputs(input1, input2, output))
    {
    }

    void initialize() override
    {
        Memory gradient1 = mOutputs[0]->getMemory();
        Memory gradient2 = mOutputs[1]->getMemory();

        for (std::size_t i = 0; i < gradient1.count(); ++i) gradient1[i] = 1.;
        for (std::size_t i = 0; i < gradient2.count(); ++i) gradient2[i] = 1.;
    }

   private:
    static std::vector<Tensor::SPtr> createOutputs(Tensor::SPtr i1,
                                                   Tensor::SPtr i2,
                                                   Tensor::SPtr out)
    {
        /* assert(i1->shape() == i2->shape() && */
        /*        i1->shape() == out->shape()); */
        Tensor::SPtr grad1 = std::make_shared<Tensor>("", i1->shape());
        Tensor::SPtr grad2 = std::make_shared<Tensor>("", i2->shape());
        return {grad1, grad2};
    }

    //! Gradients are already calculated in initialize
    void executeOper(const InputDict& inputs) override {}
};

}  // namespace

std::map<Tensor::SPtr, GradientOper::TensorMap> AddOper::gradients()
{
    std::vector<Tensor::SPtr> inputs = getInputs();
    std::vector<Tensor::SPtr> outputs = getOutputs();

    Oper::SPtr gradOper = Oper::SPtr(
        std::make_shared<AddGradientOper>(inputs[0], inputs[1], outputs[0]));
    getDefaultGraph()->insertOperation(gradOper);
    std::vector<Tensor::SPtr> grads = gradOper->getOutputs();

    return {{outputs[0], {{inputs[0], grads[0]}, {inputs[1], grads[1]}}}};
}

Tensor::SPtr add(Tensor::SPtr t1, Tensor::SPtr t2)
{
    Oper::SPtr oper = std::make_shared<AddOper>(t1, t2);
    getDefaultGraph()->insertOperation(oper);
    return oper->getOutputs()[0];
}

Tensor::SPtr operator+(Tensor::SPtr t1, Tensor::SPtr t2) { return add(t1, t2); }

}  // namespace core

ITensorSPtr add(ITensorSPtr t1, ITensorSPtr t2)
{
    core::Tensor::SPtr tensor1 = std::static_pointer_cast<core::Tensor>(t1);
    core::Tensor::SPtr tensor2 = std::static_pointer_cast<core::Tensor>(t2);
    return ITensorSPtr(core::add(tensor1, tensor2));
}

ITensorSPtr operator+(ITensorSPtr t1, ITensorSPtr t2) { return add(t1, t2); }

}  // namespace dll
