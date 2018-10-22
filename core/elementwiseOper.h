#ifndef DLL_CORE_ELEMENTWISE_OPER_H_
#define DLL_CORE_ELEMENTWISE_OPER_H_

namespace dll
{
namespace core
{

class ElementwiseOper : public Oper
{
public:
    ElementwiseOper(Tensor::SPtr t1, Tensor::SPtr t2)
        : Oper({t1, t2}, createOutputs(t1, t2))
    {}

private:
    static std::vector<Tensor::SPtr> createOutputs(Tensor::SPtr t1, Tensor::SPtr t2)
    {
        assert(t1->getTensorShape() == t2->getTensorShape());
        return {std::make_shared<Tensor>("", t1->getTensorShape())};
    }

    virtual float elementwise(float f1, float f2) = 0;

    void executeOper(const InputDict& inputs)
    {
        mInputs[0]->exec(inputs);
        mInputs[1]->exec(inputs);

        Memory input0 = mInputs[0]->getMemory();
        Memory input1 = mInputs[1]->getMemory();
        Memory output = mOutputs[0]->getMemory();

        for (std::size_t i = 0; i < output.count(); ++i)
            output[i] = elementwise(input0[i], input1[i]);
    }

} // namespace core
} // namespace dll

#endif // DLL_CORE_ELEMENTWISE_OPER_H_
