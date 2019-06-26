//! \file graphdl.h
//! \brief Main header file of GraphDL library.
//!
//! \author Adam Jędrych adam.jedrych25@gmail.com
//!

#ifndef GRAPHDL_H_
#define GRAPHDL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

//! \namespace graphdl
//!
namespace graphdl
{

//! \typedef HostTensor
//! \brief Holds memory in host.
//! \details Used for providing data to graph
//!     and for receving outputs.
//!
using HostTensor = std::vector<float>;

//! \typedef InputDict
//! \brief Map from input name to HostTensor
//!
using InputDict = std::map<std::string, HostTensor>;

//! \typedef Shape
//! \brief Represents shape of the tensor.
//!
using Shape = std::vector<unsigned int>;

//! \enum MemoryLocation
//! \brief Represents type of memory.
enum class MemoryLocation
{
    kHOST = 0,  //!< memory on host (CPU)
    kDEVICE = 1  //!< memory on device (GPU)
};

//! \brief Metatype for shared pointers.
//!
template <typename T>
using SharedPtr = std::shared_ptr<T>;

class ITensor;

//! \typedef ITensorPtr
//! \brief Shared pointer to ITensor.
//!
using ITensorPtr = SharedPtr<ITensor>;

class IGraph;

//! \typedef IGraphPtr
//! \brief Shared pointer to IGraph.
//!
using IGraphPtr = SharedPtr<IGraph>;

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
    //! \brief Returns map of all inputs to the graph.
    //!
    virtual std::map<std::string, ITensorPtr> getInputs() const = 0;

    //! \fn getWeights
    //! \brief Returns map of all weights in the graph.
    //!
    virtual std::map<std::string, ITensorPtr> getWeights() const = 0;

    virtual ~IGraph() {}
};

//! \class IInitializer
//! \brief Interface representing methods for initializing weights.
//!
class IInitializer
{
  public:
    //! \fn init
    //! \brief Initializes tensor.
    //! \param memory Pointer to memory of the tensor.
    //! \param shape Shape of the tensor.
    //! \param location Location of the tensor.
    //!
    virtual void init(float* memory, const Shape& shape,
                      MemoryLocation location) const = 0;
};

//! \typedef IInitializerPtr
//! \brief Shared pointer to IInitializer.
//!
using IInitializerPtr = SharedPtr<IInitializer>;

//! \fn IInitializerPtr constantInitializer(float value)
//! \brief Creates initializer that initializes every element to given value.
//! \param value Value to initialize elements with.
//! \return
//!
IInitializerPtr constantInitializer(float value);

//! \fn IInitializerPtr uniformInitializer(float min, float max, size_t seed)
//! \brief Creates initializer that initializes with uniform distribution.
//! \param min Minimal value of distribution.
//! \param max Maximal value of distribution.
//! \param seed Initial seed for pseudo-random generator.
//!
IInitializerPtr uniformInitializer(float min, float max, size_t seed);

//! \fn IInitializerPtr normalInitializer(float mean, float stddev, size_t seed)
//! \brief Creates initializer that initializes with normal distribution.
//! \param mean Mean of normal distribution.
//! \param stddev Standard deviation of normal distribution.
//! \param seed Initial seed for pseudo-random generator.
//!
IInitializerPtr normalInitializer(float mean, float stddev, size_t seed);

//! \fn IGraphPtr createGraph(const std::string& name)
//! \brief This function creates new IGraph object.
//! \param name Name of the graph.
//! \details If graph with the same name already exists,
//!   this will return nullptr.
//!
IGraphPtr createGraph(const std::string& name);

//! \fn void setDefaultGraph(const IGraphPtr& graph)
//! \brief This function sets graph as a default graph.
//! \param graph Graph to be set as the default.
//! \details Later all new operations will be added to this graph.
//!
void setDefaultGraph(const IGraphPtr& graph);

//! \fn IGraphPtr getDefaultGraph()
//! \brief This function returns current default graph.
//!
IGraphPtr getDefaultGraph();

//! \fn ITensorPtr createInput(const std::string& name, const Shape& shape,
//!                            MemoryLocation location)
//! \brief This function creates new input in the default graph.
//! \param name Name of the input.
//! \param shape Dimensions of the input.
//! \param location Location of the input.
//! \return Tensor representing input.
//!
ITensorPtr createInput(const std::string& name, const Shape& shape,
                       MemoryLocation location);

//! \fn ITensorPtr createWeights(const std::string& name, const Shape& shape,
//!                              const SharedPtr<IInitializer>& initializer,
//!                              MemoryLocation location)
//! \brief This function creates new weights in current graph.
//! \param name Name og the weights.
//! \param shape Dimensions of the weights.
//! \param initializer Initializer for the weights.
//! \param location Location of the weights.
//!
ITensorPtr createWeights(const std::string& name, const Shape& shape,
                         const IInitializerPtr& initializer,
                         MemoryLocation location);

//! \fn void initializeGraph()
//! \brief Initializes graph (i.e. initializes all weights).
//! \details This has to be run before evaluating anything.
//!
void initializeGraph();

//! \fn std::vector<HostTensor> eval(const std::vector<ITensorPtr>& tensors,
//!                                  const InputDict& inputs)
//! \brief Evaluates all tensors.
//! \param tensors Vector of tensors to evaluate.
//! \param inputs Map of input values.
//!
std::vector<HostTensor> eval(const std::vector<ITensorPtr>& tensors,
                             const InputDict& inputs);

//! \fn std::map<ITensorPtr, ITensorPtr> gradients(const ITensorPtr& tensor)
//! \brief Creates part of graph responsible for calculating gradients.
//! \param tensor Tensor for which gradients will be calculated (i.e. loss).
//! \return Map from weights name to tensor representing gradient.
//!
std::map<ITensorPtr, ITensorPtr> gradients(const ITensorPtr& tensor);

}  // namespace graphdl

#endif  // GRAPHDL_H_
