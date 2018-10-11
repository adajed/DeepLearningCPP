//! \file dll.h
//! \brief Main header file of dll library.
//!
//! \author Adam Jedrych adam.jedrych25@gmail.com
//!

#ifndef DLL_DL_H_
#define DLL_DL_H_

#include <string>
#include <vector>
#include <map>
#include <memory>

//! \namespace dll
namespace dll
{

class IGraph;
class ITensor;

template <typename T>
using UniquePtr = typename std::unique_ptr<T>;
template <typename T>
using SharedPtr = typename std::shared_ptr<T>;

using IGraphUPtr = SharedPtr<IGraph>;
using ITensorUPtr = SharedPtr<ITensor>;

//! \struct Tensor in host memory
//! \brief Memory used to provide inputs and to get outputs from graph.
//!
struct HostTensor
{
    const float* values;
    size_t size;
};

typedef std::map<std::string, HostTensor> InputDict;

typedef std::vector<int> Shape;

//! \class ITensor
//! \brief Interface representing tensor.
//!
class ITensor
{
public:
    //! \fn getName
    //! \brief Returns name of the tensor.
    //!
    virtual std::string getName() const = 0;

    //! \fn setName
    //! \brief Sets new name for the tensor.
    //!
    virtual void setName(std::string const& name) = 0;

    //! \fn getShape
    //! \brief Returns shape of the tensor.
    //!
    virtual Shape getShape() const = 0;

    //! \fn setShape
    //! \brief Sets new shape for the tensor, works only for input tensors.
    //!
    virtual void setShape(Shape const& shape) = 0;

    //! \fn eval
    //! \brief Evaulates tensor.
    //! \param inputs Map from names of input tensors to values.
    //!
    virtual HostTensor eval(InputDict const& inputs) = 0;
};

//! \class IGraph
//! \brief Interface representing computational graph.
//!
class IGraph
{
public:
    virtual std::string getName() const = 0;
    virtual void setName(std::string const& name) = 0;

    virtual std::vector<ITensorUPtr> getInputs() const = 0;
};

//! \fn createIGraph
//! \brief This function creates new IGraph object.
//! \param name Name of the graph.
//! If graph with the same name already exists,
//!   this will return nullptr.
//!
IGraphUPtr createGraph(std::string const& name);

//! \fn setDefaultGraph
//! \bried This function sets graph as a default graph.
//! \param graph Graph to be set as the default.
//! Later all new operations will be added to this graph.
//!
void setDefaultGraph(IGraphUPtr graph);

//! \fn getDefaultGraph
//! \brief This function returns current default graph.
//! If there is none default graph set,
//!   this will return nullptr.
//!
IGraphUPtr getDefaultGraph();

//! \fn createInput
//! \brief This function creates new input in the default graph.
//! \param name Name of the input.
//! \param shape Dimensions of the input.
//!
ITensorUPtr createInput(std::string const& name, Shape const& shape);

//! \fn createWeights
//! \brief This function creates new weights in current graph.
//! \param name Name og the weights.
//! \param shape Dimensions of the weights.
//!
ITensorUPtr createWeights(std::string const& name, Shape const& shape);

//! \fn
//! \brief Initializes graph (i.e. initializes all weights).
//! This has to be run before evaluating anything.
//!
void initializeGraph();

//! \fn eval
//! \brief Evaluates all tensors.
//! \param tensors Vector of tensors to evaluate.
//! \param inputs Map of input values.
//!
std::vector<HostTensor> eval(std::vector<ITensorUPtr> const& tensors, InputDict const& inputs);

} // namespace dll

#endif // GRAPH_DL_H_
