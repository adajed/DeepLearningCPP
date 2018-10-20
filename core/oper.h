#ifndef DLL_CORE_OPER_H_
#define DLL_CORE_OPER_H_

#include "dll.h"

namespace dll
{
namespace core
{

class Oper;

//! \class Tensor
//! \brief Implementation of ITensor interface.
//!
class Tensor : public ITensor
{
public:
    Tensor(Oper* oper, const std::string& name, const Shape& shape)
        : mName(name)
        , mShape(shape)
        , mOper(oper)
        , mOutputOps()
        , mIsEvaluated(false)
    {}

    std::string getName() const override;
    void setName(const std::string& name) override;

    Shape getShape() const override;
    void setShape(const Shape& shape) override;

    HostTensor eval(const InputDict& inputs) override;

    void setHostTensor(HostTensor tensor);

    void exec(const InputDict& inputs);

    void reset();

    ~Tensor();

private:
    std::string mName; //!< Tensor name.
    Shape mShape; //< Tensor shape.

    Oper* mOper;
    std::vector<Oper*> mOutputOps;
    bool mIsEvaluated;
};

inline Tensor* createTensor(Oper* oper, const std::string& name, const Shape& shape)
{
    return new Tensor(oper, name, shape);
}

class Oper
{
public:
    Oper(const std::vector<Tensor*>& inputs, std::vector<Tensor*> outputs)
        : mIsEvaluated(false), mInputs(inputs), mOutputs(outputs)
    {}

    std::vector<Tensor*> getInputs();

    std::vector<Tensor*> getOutputs();

    virtual void exec(const InputDict& inputs);

    virtual bool supportsCPU() { return false; }
    virtual bool supportsGPU() { return false; }

    void reset();

    virtual ~Oper()
    {
        for (Tensor* tensor : mOutputs)
            delete tensor;
    }

private:
    virtual void executeOper(const InputDict& inputs) = 0;

    bool mIsEvaluated;

protected:
    std::vector<Tensor*> mInputs;
    std::vector<Tensor*> mOutputs;
};

} // namespace core
} // namespace dll

#endif // DLL_CORE_OPER_H_
