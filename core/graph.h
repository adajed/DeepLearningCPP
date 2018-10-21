#ifndef GRAPHDL_CORE_GRAPH_H_
#define GRAPHDL_CORE_GRAPH_H_

#include "dll.h"
#include "oper.h"
#include "input.h"

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
    Graph(std::string const& name);

    std::string getName() const override;
    void setName(std::string const& name) override;

    //! \fn getInputs
    //! \brief Returns all the inputs to the graph.
    //!
    std::vector<ITensor*> getInputs() const override;

    //! \fn reset
    //! \brief Prepares graph for next computation.
    //!
    void reset();

    //! \fn allocateMemory
    //! \brief Allocates memory for all tensors in the graph.
    //!
    bool allocateMemory();

    //! \fn freeMemory
    //! \brief Frees memory for all tensors in te graph.
    //!
    bool freeMemory();

    //! \fn addInput
    //! \brief Adds new input to the graph.
    //! \param name Name of the new input.
    //! \param shape Shape of the new input.
    //!
    //! \return Pointer to tensor representing new input.
    Tensor* addInput(const std::string& name, const Shape& shape);

    //! \fn addWeights
    //! \brief Adds new weights to the graph.
    //! \param name Name of new weights.
    //! \param shape Shape of new weights.
    //!
    //! \return Pointer to tensor representing new weights.
    Tensor* addWeights(const std::string& name, const Shape& shape);

    ~Graph()
    {
        for (std::pair<std::string, InputOper*> pair : mInputOps)
            delete pair.second;
    }

private:
    void addOper(Oper* oper);

    std::string mName; //!< Name of the graph.
    std::map<std::string, InputOper*> mInputOps;

    std::map<Oper::ID, Oper*> mOps;
    std::map<Tensor::ID, Tensor*> mTensors;
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
        Graph* graph = new Graph(DEFAULT_GRAPH_NAME);
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
    Graph* at(std::string const& name);

    //! \fn insert
    //! \brief Registers new graph.
    //! If there is already a graph with the same name,
    //!     this will return false and won't change the register.
    //!
    bool insert(Graph* graph);

    //! \fn getDefaultGraph
    //! \brief Returns the default graph.
    //!
    Graph* getDefaultGraph();

    //! \fn setDefaultGraph
    //! \brief Sets given graph as a default one.
    //! If given graph is not registered yet, this will
    //!     register it and then set is as a deafult graph.
    //!
    void setDefaultGraph(Graph*);

    void clear();

    ~GraphRegister()
    {
        for (std::pair<std::string, Graph*> pair : mGraphDict)
            delete pair.second;
    }

private:
    std::map<std::string, Graph*> mGraphDict;
    Graph* mDefaultGraph;
};

} // namespace core
} // namespace dll

#endif // GRAPHDL_CORE_GRAPH_H_
