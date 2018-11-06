//! \file graphdl.h
//! \brief Main header file of GraphDL library.
//!
//! \author Adam Jedrych adam.jedrych25@gmail.com
//!

#ifndef GRAPHDL_H_
#define GRAPHDL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

//! \namespace graphdl
namespace graphdl
{
using HostTensor = std::vector<float>;
using InputDict = std::map<std::string, HostTensor>;
using Shape = std::vector<unsigned int>;

class ITensor;
using ITensorPtr = std::shared_ptr<ITensor>;

class IGraph;
using IGraphPtr = std::shared_ptr<IGraph>;

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
    virtual void setName(const std::string& name) = 0;

    //! \fn getShape
    //! \brief Returns shape of the tensor.
    //!
    virtual Shape getShape() const = 0;

    //! \fn eval
    //! \brief Evaulates tensor.
    //! \param inputs Map from names of input tensors to values.
    //!
    virtual HostTensor eval(const InputDict& inputs) = 0;

    virtual ~ITensor() {}
};

//! \class IGraph
//! \brief Interface representing computational graph.
//!
class IGraph
{
  public:
    //! \fn getName
    //! \brief Returns name of the graph.
    //!
    virtual std::string getName() const = 0;

    //! \fn setName
    //! \brief Sets new name for the graph.
    //!
    virtual void setName(const std::string& name) = 0;

    //! \fn getInputs
    //! \brief Returns map of all inputs to the graph
    //!
    virtual std::map<std::string, ITensorPtr> getInputs() const = 0;

    //! \fn getWeights
    //! \brief Returns map of all weights in the graph
    //!
    virtual std::map<std::string, ITensorPtr> getWeights() const = 0;

    virtual ~IGraph() {}
};

//! \fn createIGraph
//! \brief This function creates new IGraph object.
//! \param name Name of the graph.
//! If graph with the same name already exists,
//!   this will return nullptr.
//!
IGraphPtr createGraph(const std::string& name);

//! \fn setDefaultGraph
//! \bried This function sets graph as a default graph.
//! \param graph Graph to be set as the default.
//! Later all new operations will be added to this graph.
//!
void setDefaultGraph(IGraphPtr graph);

//! \fn getDefaultGraph
//! \brief This function returns current default graph.
//!
IGraphPtr getDefaultGraph();

//! \fn createInput
//! \brief This function creates new input in the default graph.
//! \param name Name of the input.
//! \param shape Dimensions of the input.
//!
ITensorPtr createInput(const std::string& name, const Shape& shape);

//! \fn createWeights
//! \brief This function creates new weights in current graph.
//! \param name Name og the weights.
//! \param shape Dimensions of the weights.
//!
ITensorPtr createWeights(const std::string& name, const Shape& shape);

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
std::vector<HostTensor> eval(const std::vector<ITensorPtr>& tensors,
                             const InputDict& inputs);

//! \fn gradients
//! \brief Creates part of graph responsible for calculating gradients.
//! \param tensor Tensor for which gradients will be calculated (i.e. loss).
//! \return Map from weights name to tensor representing gradient.
//!
std::map<ITensorPtr, ITensorPtr> gradients(ITensorPtr tensor);

}  // namespace graphdl

#endif  // GRAPHDL_H_
