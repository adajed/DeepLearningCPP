#ifndef DLL_CORE_OPER_H_
#define DLL_CORE_OPER_H_

#include "dll.h"
#include "memory.h"
#include "tensorShape.h"

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
    using ID = std::size_t;

    Tensor(Oper* oper, const std::string& name, const TensorShape& shape)
        : mID(nextID())
        , mName(name)
        , mShape(shape)
        , mOper(oper)
        , mOutputOps()
        , mIsEvaluated(false)
        , mMemory(MemoryType::kHOST_MEMORY, shape.count())
    {}

    ID getID() const;

    std::string getName() const override;
    void setName(const std::string& name) override;

    Shape getShape() const override;
    void setShape(const Shape& shape) override;

    TensorShape getTensorShape() const;
    void setTensorShape(const TensorShape& shape);

    Memory getMemory();

    bool allocateMemory();

    void freeMemory();

    void eval(const InputDict& inputs, HostTensor* hostTensor) override;

    void exec(const InputDict& inputs);

    void reset();

    ~Tensor();

private:
    static ID nextID()
    {
        static ID idCounter = 0;
        return idCounter++;
    }

    ID mID;
    std::string mName; //!< Tensor name.
    TensorShape mShape; //< Tensor shape.

    Oper* mOper;
    std::vector<Oper*> mOutputOps;
    bool mIsEvaluated;

    Memory mMemory;
};

inline Tensor* createTensor(Oper* oper, const std::string& name, const Shape& shape)
{
    return new Tensor(oper, name, shape);
}

class Oper
{
public:
    using ID = std::size_t;

    Oper(const std::vector<Tensor*>& inputs, std::vector<Tensor*> outputs)
        : mID(nextID()), mIsEvaluated(false), mInputs(inputs), mOutputs(outputs)
    {}

    ID getID() const;
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

    static ID nextID()
    {
        static ID idCounter = 0;
        return idCounter++;
    }

    ID mID;
    bool mIsEvaluated;

protected:
    std::vector<Tensor*> mInputs;
    std::vector<Tensor*> mOutputs;
};

} // namespace core
} // namespace dll

#endif // DLL_CORE_OPER_H_
