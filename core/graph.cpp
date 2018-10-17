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
TensorSPtr Graph::addInput(const std::string& name, const Shape& shape)
{
    return TensorSPtr(nullptr);
}

// TODO: Graph::addWeights
TensorSPtr Graph::addWeights(const std::string& name, const Shape& shape)
{
    return TensorSPtr(nullptr);
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

GraphSPtr GraphRegister::at(const std::string& name)
{
    return mGraphDict.at(name);
}

bool GraphRegister::insert(GraphSPtr graph)
{
    if (hasKey(graph->getName()))
        return false;
    mGraphDict[graph->getName()] = graph;
    return true;
}

GraphSPtr GraphRegister::getDefaultGraph()
{
    return mDefaultGraph;
}

void GraphRegister::setDefaultGraph(GraphSPtr graph)
{
    mDefaultGraph = graph;
}

void GraphRegister::clear()
{
    GraphSPtr default_graph = mGraphDict.at(GraphRegister::DEFAULT_GRAPH_NAME);
    mGraphDict.clear();
    mGraphDict[GraphRegister::DEFAULT_GRAPH_NAME] = default_graph;
}

} // namespace core

#define GLOBAL_REGISTER core::GraphRegister::getGlobalGraphRegister()

IGraphUPtr createGraph(const std::string& name)
{
    if (GLOBAL_REGISTER.hasKey(name))
        return nullptr;

    core::GraphSPtr graph = std::make_shared<core::Graph>(name);
    GLOBAL_REGISTER.insert(graph);
    return graph;
}

void setDefaultGraph(IGraphUPtr graph)
{
    core::GraphSPtr g = std::static_pointer_cast<core::Graph>(graph);
    GLOBAL_REGISTER.setDefaultGraph(g);
}

IGraphUPtr getDefaultGraph()
{
    return GLOBAL_REGISTER.getDefaultGraph();
}

ITensorUPtr createInput(const std::string& name, const Shape& shape)
{
    core::GraphSPtr graph = GLOBAL_REGISTER.getDefaultGraph();
    return graph->addInput(name, shape);
}

ITensorUPtr createWeights(const std::string& name, const Shape& shape)
{
    core::GraphSPtr graph = GLOBAL_REGISTER.getDefaultGraph();
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

