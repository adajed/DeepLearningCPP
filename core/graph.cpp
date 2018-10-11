#include "dll.h"
#include "graph.h"

namespace dll
{
namespace core
{

const std::string GraphRegister::DEFAULT_GRAPH_NAME = "default_graph";

std::string Graph::getName() const
{
    return mName;
}

void Graph::setName(const std::string& name)
{
    mName = name;
}


// TODO: Graph::getInputs
std::vector<ITensorUPtr> Graph::getInputs() const
{
    return {};
}

// TODO: Graph::reset
void Graph::reset()
{
}

// TODO: Graph::allocateMemory
bool Graph::allocateMemory()
{
    return false;
}

// TODO: Graph::freeMemory
bool Graph::freeMemory()
{
    return false;
}


// TODO: Graph::addInput
TensorUPtr Graph::addInput(const std::string& name, const Shape& shape)
{
    return TensorUPtr(nullptr);
}

// TODO: Graph::addWeights
TensorUPtr Graph::addWeights(const std::string& name, const Shape& shape)
{
    return TensorUPtr(nullptr);
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

GraphUPtr GraphRegister::at(const std::string& name)
{
    return mGraphDict.at(name);
}

bool GraphRegister::insert(GraphUPtr graph)
{
    if (hasKey(graph->getName()))
        return false;
    mGraphDict[graph->getName()] = graph;
    return true;
}

GraphUPtr GraphRegister::getDefaultGraph()
{
    return mDefaultGraph;
}

void GraphRegister::setDefaultGraph(GraphUPtr graph)
{
    mDefaultGraph = graph;
}

} // namespace core

IGraphUPtr getDefaultGraph()
{
    return core::GraphRegister::getGlobalGraphRegister().getDefaultGraph();
}

ITensorUPtr createInput(const std::string& name, const Shape& shape)
{
    core::GraphUPtr graph = core::GraphRegister::getGlobalGraphRegister().getDefaultGraph();
    return graph->addInput(name, shape);
}

ITensorUPtr createWeights(const std::string& name, const Shape& shape)
{
    core::GraphUPtr graph = core::GraphRegister::getGlobalGraphRegister().getDefaultGraph();
    return graph->addWeights(name, shape);
}

// TODO: initializeGraph
void initializeGraph()
{
}

std::vector<HostTensor> eval(std::vector<ITensorUPtr> const& tensors, InputDict const& inputs)
{
    return {};
}

} // namespace dll

