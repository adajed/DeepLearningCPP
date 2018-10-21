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

//! \struct Tensor in host memory
//! \brief Memory used to provide inputs and to get outputs from graph.
//!
struct HostTensor
{
    float* values;
    size_t count;
};

typedef std::map<std::string, HostTensor> InputDict;

typedef std::vector<unsigned int> Shape;

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
    virtual void eval(InputDict const& inputs, HostTensor* hostTensor) = 0;

    virtual ~ITensor() {}
};

//! \class IGraph
//! \brief Interface representing computational graph.
//!
class IGraph
{
public:
    virtual std::string getName() const = 0;
    virtual void setName(std::string const& name) = 0;

    virtual std::vector<ITensor*> getInputs() const = 0;

    virtual ~IGraph() {}
};

//! \fn createIGraph
//! \brief This function creates new IGraph object.
//! \param name Name of the graph.
//! If graph with the same name already exists,
//!   this will return nullptr.
//!
IGraph* createGraph(std::string const& name);

//! \fn setDefaultGraph
//! \bried This function sets graph as a default graph.
//! \param graph Graph to be set as the default.
//! Later all new operations will be added to this graph.
//!
void setDefaultGraph(IGraph* graph);

//! \fn getDefaultGraph
//! \brief This function returns current default graph.
//!
IGraph* getDefaultGraph();

//! \fn createInput
//! \brief This function creates new input in the default graph.
//! \param name Name of the input.
//! \param shape Dimensions of the input.
//!
ITensor* createInput(std::string const& name, Shape const& shape);

//! \fn createWeights
//! \brief This function creates new weights in current graph.
//! \param name Name og the weights.
//! \param shape Dimensions of the weights.
//!
ITensor* createWeights(std::string const& name, Shape const& shape);

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
void eval(std::vector<ITensor*> const& tensors, InputDict const& inputs, std::vector<HostTensor*> hostTensors);

} // namespace dll

#endif // GRAPH_DL_H_
