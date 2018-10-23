#ifndef GRAPHDL_CORE_GRAPH_H_
#define GRAPHDL_CORE_GRAPH_H_

#include "dll.h"
#include "oper.h"
#include "input.h"
#include "weights.h"

namespace dll
{
namespace core
{

class Graph;

//! \class Graph
//! \brief Implementation of IGraph interface.
//!
class Graph : public IGraph
{
public:
    using UPtr = std::shared_ptr<Graph>;
    using SPtr = std::shared_ptr<Graph>;
    using WeakPtr = std::weak_ptr<Graph>;

    Graph(std::string const& name);

    std::string getName() const override;
    void setName(std::string const& name) override;

    //! \fn getInputs
    //! \brief Returns all the inputs to the graph.
    //!
    std::map<std::string, ITensorSPtr> getInputs() const override;

    //! \fn getWeights
    //! \brief Returns all the weights in the graph.
    //!
    std::map<std::string, ITensorSPtr> getWeights() const override;

    //! \fn reset
    //! \brief Prepares graph for next computation.
    //!
    void reset();

    //! \fn allocateMemory
    //! \brief Allocates memory for all tensors in the graph.
    //!
    bool allocateMemory();

    //! \fn initializeWeights
    //! \brief Initializes all weights in the graph.
    //!
    bool initializeWeights();

    //! \fn freeMemory
    //! \brief Frees memory for all tensors in te graph.
    //!
    void freeMemory();

    //! \fn addInput
    //! \brief Adds new input to the graph.
    //! \param name Name of the new input.
    //! \param shape Shape of the new input.
    //!
    //! \return Pointer to tensor representing new input.
    Tensor::SPtr addInput(const std::string& name, const Shape& shape);

    //! \fn addWeights
    //! \brief Adds new weights to the graph.
    //! \param name Name of new weights.
    //! \param shape Shape of new weights.
    //!
    //! \return Pointer to tensor representing new weights.
    Tensor::SPtr addWeights(const std::string& name, const Shape& shape);

    //! \fn insertOperation
    //! \brief Adds operation and all its tensors to the graph.
    //! \param oper Shared pointer to oper that will be added to the graph.
    //!
    void insertOperation(Oper::SPtr oper);

private:
    std::string mName; //!< Name of the graph.
    std::map<std::string, std::shared_ptr<InputOper>> mInputOps; //!< Map of input ops.
    std::map<std::string, std::shared_ptr<WeightsOper>> mWeightsOps; //!< Map of weights ops.

    std::map<Oper::ID, Oper::SPtr> mOps; //!< Map of all ops inside the graph.
    std::map<Tensor::ID, Tensor::SPtr> mTensors; //!< MAp of all tensors inside the graph.
};

//! \class GraphRegister
//! \brief Register with all created graphs.
//!
class GraphRegister
{
private:
    // name of initial default graph
    static const std::string DEFAULT_GRAPH_NAME;

    static GraphRegister globalRegister;

    // private constructor to prevent creation of new registers
    GraphRegister() : mGraphDict()
    {
        // create initial graph and set it as the default
        Graph::SPtr graph = std::make_shared<Graph>(DEFAULT_GRAPH_NAME);
        mGraphDict[DEFAULT_GRAPH_NAME] = graph;
        mDefaultGraph = graph;
    }

public:
    //! \fn getGlobalIGraphRegister
    //! \brief Returns the global register of graphs.
    //!
    static GraphRegister& getGlobalGraphRegister();

    //! \fn hasKey
    //! \brief Checks whether graph with given name already exists.
    //!
    bool hasKey(std::string const& name) const;

    //! \fn at
    //! \brief Returns graph with given name.
    //!
    Graph::SPtr at(std::string const& name);

    //! \fn insert
    //! \brief Registers new graph.
    //! If there is already a graph with the same name,
    //!     this will return false and won't change the register.
    //!
    bool insert(Graph::SPtr graph);

    //! \fn getDefaultGraph
    //! \brief Returns the default graph.
    //!
    Graph::SPtr getDefaultGraph();

    //! \fn setDefaultGraph
    //! \brief Sets given graph as a default one.
    //! If given graph is not registered yet, this will
    //!     register it and then set is as a deafult graph.
    //!
    void setDefaultGraph(Graph::SPtr graph);

    void clear();

private:
    std::map<std::string, Graph::SPtr> mGraphDict;
    Graph::SPtr mDefaultGraph;
};

Graph::SPtr getDefaultGraph();

} // namespace core
} // namespace dll

#endif // GRAPHDL_CORE_GRAPH_H_
