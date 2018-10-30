#include <assert.h>

#include "graph.h"

namespace graphdl
{
namespace core
{
const std::string GraphRegister::DEFAULT_GRAPH_NAME = "default_graph";

Graph::Graph(const std::string& name)
    : mName(name), mInputLayers(), mWeightLayers(), mLayers(), mTensors()
{
}

std::string Graph::getName() const { return mName; }

void Graph::setName(const std::string& name) { mName = name; }

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
        weightTensors.insert({weight->getName(), weight});
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

Tensor::SPtr Graph::addInput(const std::string& name, const Shape& shape)
{
    if (mInputLayers.count(name) > 0) return nullptr;

    std::shared_ptr<InputLayer> layer =
        std::make_shared<InputLayer>(name, shape);
    mInputLayers.insert({name, layer});
    insertLayer(Layer::SPtr(layer));

    return layer->getOutputs()[0];
}

Tensor::SPtr Graph::addWeights(const std::string& name, const Shape& shape)
{
    if (mWeightLayers.count(name) > 0) return nullptr;

    auto weightsLayer = std::make_shared<WeightsLayer>(name, shape);
    mWeightLayers.insert({name, weightsLayer});
    insertLayer(Layer::SPtr(weightsLayer));

    return weightsLayer->getOutputs()[0];
}

void Graph::insertLayer(Layer::SPtr layer)
{
    Layer::ID opID = layer->getID();
    mLayers.insert({opID, layer});
    for (Tensor::SPtr tensor : layer->getOutputs())
    {
        Tensor::ID tensorID = tensor->getID();
        tensor->setLayer(layer);
        mTensors.insert({tensorID, tensor});
    }
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

Graph::SPtr GraphRegister::getDefaultGraph() { return mDefaultGraph; }

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
