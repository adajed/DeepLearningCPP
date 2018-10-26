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
    using UPtr = std::unique_ptr<Tensor>;
    using SPtr = std::shared_ptr<Tensor>;
    using WeakPtr = std::weak_ptr<Tensor>;

    Tensor(const std::string& name, const TensorShape& shape)
        : mID(nextID()),
          mName(name),
          mShape(shape),
          mOper(),
          mOutputOps(),
          mIsEvaluated(false),
          mMemory(MemoryType::kHOST_MEMORY, shape.count())
    {
    }

    ID getID() const;

    std::string getName() const override;
    void setName(const std::string& name) override;

    Shape getShape() const override;
    void setShape(const Shape& shape) override;

    TensorShape shape() const;
    void setTensorShape(const TensorShape& shape);

    std::shared_ptr<Oper> getOper() const;
    void setOper(std::shared_ptr<Oper> oper);

    Memory getMemory();

    bool allocateMemory();

    void freeMemory();

    void eval(const InputDict& inputs, HostTensor hostTensor) override;

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
    std::string mName;   //!< Tensor name.
    TensorShape mShape;  //< Tensor shape.

    std::weak_ptr<Oper> mOper;
    std::vector<std::shared_ptr<Oper>> mOutputOps;
    bool mIsEvaluated;

    Memory mMemory;
};

Tensor::SPtr createTensor(const std::string& name, const TensorShape& shape);

class Oper
{
   public:
    using ID = std::size_t;
    using UPtr = std::unique_ptr<Oper>;
    using SPtr = std::shared_ptr<Oper>;
    using WeakPtr = std::weak_ptr<Oper>;

    Oper(const std::vector<Tensor::SPtr>& inputs,
         std::vector<Tensor::SPtr> outputs)
        : mID(nextID()), mIsEvaluated(false), mInputs(), mOutputs(outputs)
    {
        for (Tensor::SPtr input : inputs)
            mInputs.push_back(Tensor::WeakPtr(input));
    }

    ID getID() const;
    std::vector<Tensor::SPtr> getInputs();
    std::vector<Tensor::SPtr> getOutputs();

    virtual void exec(const InputDict& inputs);

    virtual bool hasGradient() const { return false; }

    virtual void initialize() {}

    void reset();

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
    std::vector<Tensor::WeakPtr> mInputs;
    std::vector<Tensor::SPtr> mOutputs;
};

}  // namespace core
}  // namespace dll

#endif  // DLL_CORE_OPER_H_
