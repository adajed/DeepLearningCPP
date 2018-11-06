#include "graph.h"

#include <assert.h>

namespace graphdl
{
namespace core
{
const std::string GraphRegister::DEFAULT_GRAPH_NAME = "default_graph";

Graph::Graph(const std::string& name)
    : mName(name),
      mInputLayers(),
      mWeightLayers(),
      mLayers(),
      mTensors(),
      mTensorCounter(0),
      mLayerCounter(0)
{
}

std::string Graph::getName() const
{
    return mName;
}

void Graph::setName(const std::string& name)
{
    mName = name;
}

std::map<std::string, Tensor::SPtr> Graph::getInputs() const
{
    std::map<std::string, Tensor::SPtr> inputTensors;
    for (auto input : mInputLayers)
    {
        Tensor::SPtr tensor = input->getOutputs()[0];
        inputTensors.insert({tensor->getName(), tensor});
    }
    return inputTensors;
}

std::map<std::string, Tensor::SPtr> Graph::getWeights() const
{
    std::map<std::string, Tensor::SPtr> weightTensors;
    for (auto weight : mWeightLayers)
    {
        Tensor::SPtr tensor = weight->getOutputs()[0];
        weightTensors.insert({tensor->getName(), tensor});
    }
    return weightTensors;
}

void Graph::prepareForNextComputation()
{
    for (auto pair : mLayers) pair.second->reset();
}

bool Graph::allocateMemory()
{
    std::vector<Tensor::SPtr> allocatedTensors;
    for (std::pair<Tensor::ID, Tensor::SPtr> pair : mTensors)
    {
        if (!pair.second->allocateMemory())
        {
            for (Tensor::SPtr t : allocatedTensors) t->freeMemory();
            return false;
        }
        allocatedTensors.push_back(pair.second);
    }
    return true;
}

void Graph::initializeLayers()
{
    for (auto pair : mLayers) pair.second->initialize();
}

void Graph::freeMemory()
{
    for (auto pair : mTensors) pair.second->freeMemory();
}

Tensor::SPtr Graph::addInput(const std::string& name, Layer::SPtr layer)
{
    mInputLayers.push_back(layer);
    return layer->getOutputs()[0];
}

Tensor::SPtr Graph::addWeights(const std::string& name, Layer::SPtr layer)
{
    mWeightLayers.push_back(layer);
    return layer->getOutputs()[0];
}

void Graph::insertLayer(Layer::SPtr layer)
{
    Layer::ID opID = layer->getID();
    mLayers.insert({opID, layer});
}

void Graph::insertTensor(Tensor::SPtr tensor)
{
    Tensor::ID tensorID = tensor->getID();
    mTensors.insert({tensorID, tensor});
}

Tensor::ID Graph::nextTensorID()
{
    return mTensorCounter++;
}

Layer::ID Graph::nextLayerID()
{
    return mLayerCounter++;
}

GraphRegister& GraphRegister::getGlobalGraphRegister()
{
    static GraphRegister gRegister;
    return gRegister;
}

bool GraphRegister::hasKey(const std::string& name) const
{
    return mGraphDict.count(name);
}

Graph::SPtr GraphRegister::at(const std::string& name)
{
    return mGraphDict.at(name);
}

bool GraphRegister::insert(Graph::SPtr graph)
{
    if (hasKey(graph->getName())) return false;
    mGraphDict[graph->getName()] = graph;
    return true;
}

Graph::SPtr GraphRegister::getDefaultGraph()
{
    return mDefaultGraph;
}

void GraphRegister::setDefaultGraph(Graph::SPtr graph)
{
    mDefaultGraph = graph;
}

void GraphRegister::clear()
{
    mGraphDict.clear();
    Graph::SPtr default_graph =
        std::make_shared<Graph>(GraphRegister::DEFAULT_GRAPH_NAME);
    mGraphDict[GraphRegister::DEFAULT_GRAPH_NAME] = default_graph;
    mDefaultGraph = default_graph;
}

GraphRegister& getGraphRegister()
{
    return GraphRegister::getGlobalGraphRegister();
}

Graph::SPtr getDefaultGraph()
{
    return GraphRegister::getGlobalGraphRegister().getDefaultGraph();
}

}  // namespace core
}  // namespace graphdl
