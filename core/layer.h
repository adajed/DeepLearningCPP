#ifndef GRAPHDL_CORE_LAYER_H_
#define GRAPHDL_CORE_LAYER_H_

#include "graphdl.h"
#include "memory.h"
#include "tensorShape.h"

#include <exception>

namespace graphdl
{
namespace core
{
class Graph;
class Layer;

//! \class Tensor
//! \brief Implementation of tensor inside the graph.
//!
class Tensor
{
  public:
    using ID = std::size_t;
    using UPtr = std::unique_ptr<Tensor>;
    using SPtr = std::shared_ptr<Tensor>;
    using WeakPtr = std::weak_ptr<Tensor>;

    Tensor(ID id, const std::string& name, const TensorShape& shape);

    //! \fn getID
    //!
    ID getID() const;

    //! \fn getName
    //!
    std::string getName() const;

    //! \fn setName
    //!
    void setName(const std::string& name);

    //! \fn getShape
    //!
    TensorShape getShape() const;

    //! \fn getOper
    //! \brief Return layer which is producing this tensor.
    //!
    std::shared_ptr<Layer> getLayer() const;

    //! \fn setLayer
    //!
    void setLayer(std::shared_ptr<Layer> layer);

    //! \fn getGraph
    //! \brief Returns graph to which this tensor belongs to.
    //!
    std::shared_ptr<Graph> getGraph() const;

    //! \fn getMemory
    //!
    Memory getMemory();

    //! \fn allocateMemory
    //! \brief Allocates tensor memory for a computation.
    //! This runs only once during graph initialization.
    //!
    bool allocateMemory();

    //! \fn freeMemory
    //!
    void freeMemory();

    //! \fn eval
    //! \brief Evaluates tensor, recursive.
    //!
    void eval(const InputDict& inputs);

    //! \fn reset
    //! \brief Resets tensor before next graph evaulation.
    //!
    void reset();

    ~Tensor();

  private:
    ID mID;
    std::string mName;  //!< Tensor name.
    TensorShape mShape;  //< Tensor shape.

    bool mIsEvaluated;
    std::weak_ptr<Layer> mLayer;
    Memory mMemory;
};

Tensor::SPtr createTensor(const std::string& name, const TensorShape& shape);

//! \class Layer
//! \brief Class representing part of computation.
//!
class Layer
{
  public:
    using ID = std::size_t;
    using UPtr = std::unique_ptr<Layer>;
    using SPtr = std::shared_ptr<Layer>;
    using WeakPtr = std::weak_ptr<Layer>;
    using TensorMap = std::map<Tensor::SPtr, Tensor::SPtr>;

    Layer(ID id, const std::vector<Tensor::SPtr>& inputs,
          const std::vector<Tensor::SPtr>& outputs);

    //! \fn getID
    //!
    ID getID() const;

    //! \fn getGraph
    //!
    std::shared_ptr<Graph> getGraph() const;

    //! \fn setGraph
    //!
    void setGraph(std::shared_ptr<Graph> graph);

    //! \fn getInputs
    //!
    std::vector<Tensor::SPtr> getInputs();

    //! \fn getOutputs
    //!
    std::vector<Tensor::SPtr> getOutputs();

    //! \fn eval
    //! \brief Evaluates layer, recursive.
    //!
    virtual void eval(const InputDict& inputs);

    //! \fn hasGradient
    //!
    virtual bool hasGradient() const { return false; }

    //! \fn gradients
    //! \brief Returns gradients of this tensor with respect to weights.
    //!
    virtual TensorMap gradients(Tensor::SPtr output, Tensor::SPtr outputGrad)
    {
        throw std::runtime_error("Gradients not implemented");
    }

    //! \fn initialize
    //! \brief Initializes layer before computation process.
    //! This will be run once during graph initialization.
    //! It gives layer chance to prepare before running.
    //!
    virtual void initialize() {}

    //! \fn reset
    //! \brief Resets layer before next graph evaulation.
    //!
    void reset();

  private:
    //! \fn executeOper
    //!
    virtual void execute(const InputDict& inputs) = 0;

    ID mID;
    bool mIsEvaluated;
    std::weak_ptr<Graph> mGraph;

  protected:
    std::vector<Tensor::WeakPtr> mInputs;
    std::vector<Tensor::SPtr> mOutputs;
};

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYER_H_
