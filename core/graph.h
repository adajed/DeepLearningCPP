#ifndef GRAPHDL_CORE_GRAPH_H_
#define GRAPHDL_CORE_GRAPH_H_

#include "input.h"
#include "layer.h"
#include "weights.h"
#include "weightsNamespaces.h"

namespace graphdl
{
namespace core
{
//! \class Graph
//! \brief Class representing computation graph.
//!
class Graph
{
  public:
    using UPtr = std::shared_ptr<Graph>;
    using SPtr = std::shared_ptr<Graph>;
    using WeakPtr = std::weak_ptr<Graph>;

    Graph(std::string name);

    std::string getName() const;
    void setName(const std::string& name);

    //! \fn getInputs
    //! \brief Returns all the inputs to the graph.
    //!
    std::map<std::string, Tensor::SPtr> getInputs() const;

    //! \fn getWeights
    //! \brief Returns all the weights in the graph.
    //!
    std::map<std::string, Tensor::SPtr> getWeights() const;

    //! \fn prepareForNextComputation
    //! \brief Prepares graph for next computation.
    //!
    void prepareForNextComputation();

    //! \fn allocateMemory
    //! \brief Allocates memory for all tensors in the graph.
    //!
    bool allocateMemory();

    //! \fn initializeLayers
    //! \brief Initializes all operations in the graph.
    //!
    void initializeLayers();

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
    Tensor::SPtr addInput(const std::string& name, const Layer::SPtr& layer);

    //! \fn addWeights
    //! \brief Adds new weights to the graph.
    //! \param name Name of new weights.
    //! \param shape Shape of new weights.
    //!
    //! \return Pointer to tensor representing new weights.
    Tensor::SPtr addWeights(const std::string& name, const Layer::SPtr& layer,
                            const std::string& nspace);

    //! \fn insertLayer
    //!
    void insertLayer(Layer::SPtr layer);

    //! \fn insertTensor
    //!
    void insertTensor(Tensor::SPtr tensor);

    Tensor::ID nextTensorID();

    Layer::ID nextLayerID();

    const WeightsNamespaces& getWeightsNamespaces() const;

  private:
    std::string mName;  //!< Name of the graph.
    std::vector<Layer::SPtr> mInputLayers;
    WeightsNamespaces mWeights;
    std::map<Layer::ID, Layer::SPtr> mLayers;
    std::map<Tensor::ID, Tensor::SPtr> mTensors;
    Tensor::ID mTensorCounter;
    Layer::ID mLayerCounter;
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
    bool hasKey(const std::string& name) const;

    //! \fn at
    //! \brief Returns graph with given name.
    //!
    Graph::SPtr at(const std::string& name);

    //! \fn insert
    //! \brief Registers new graph.
    //! If there is already a graph with the same name,
    //!     this will return false and won't change the register.
    //!
    bool insert(const Graph::SPtr& graph);

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

GraphRegister& getGraphRegister();

Graph::SPtr getDefaultGraph();

template <typename LayerType, typename... Args>
Layer::SPtr createLayer(Args... args)
{
    core::Graph::SPtr graph = core::getDefaultGraph();
    Layer::SPtr layer =
        std::make_shared<LayerType>(graph->nextLayerID(), args...);
    layer->setGraph(graph);
    graph->insertLayer(layer);
    for (Tensor::SPtr tensor : layer->getOutputs())
    {
        tensor->setLayer(layer);
        graph->insertTensor(tensor);
    }
    return layer;
}

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_GRAPH_H_
