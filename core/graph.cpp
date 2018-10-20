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


std::vector<ITensor*> Graph::getInputs() const
{
    std::vector<ITensor*> inputs;
    for (std::pair<std::string, InputOper*> op : mInputOps)
        inputs.push_back(op.second->getOutputs()[0]);

    return inputs;
}

void Graph::reset()
{
    for (std::pair<std::string, InputOper*> op : mInputOps)
        op.second->reset();
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


Tensor* Graph::addInput(const std::string& name, const Shape& shape)
{
    if (mInputOps.count(name) > 0)
        return nullptr;

    InputOper* oper = new InputOper(name, shape);
    mInputOps[name] = oper;
    return oper->getOutputs()[0];
}

// TODO: Graph::addWeights
Tensor* Graph::addWeights(const std::string& name, const Shape& shape)
{
    return nullptr;
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

Graph* GraphRegister::at(const std::string& name)
{
    return mGraphDict.at(name);
}

bool GraphRegister::insert(Graph* graph)
{
    if (hasKey(graph->getName()))
        return false;
    mGraphDict[graph->getName()] = graph;
    return true;
}

Graph* GraphRegister::getDefaultGraph()
{
    return mDefaultGraph;
}

void GraphRegister::setDefaultGraph(Graph* graph)
{
    mDefaultGraph = graph;
}

void GraphRegister::clear()
{
    for (std::pair<std::string, Graph*> pair : mGraphDict)
        delete pair.second;
    mGraphDict.clear();
    mGraphDict[GraphRegister::DEFAULT_GRAPH_NAME] = new Graph(GraphRegister::DEFAULT_GRAPH_NAME);
}

} // namespace core

#define GLOBAL_REGISTER core::GraphRegister::getGlobalGraphRegister()

IGraph* createGraph(const std::string& name)
{
    if (GLOBAL_REGISTER.hasKey(name))
        return nullptr;

    core::Graph* graph = new core::Graph(name);
    GLOBAL_REGISTER.insert(graph);
    return graph;
}

void setDefaultGraph(IGraph* graph)
{
    core::Graph* g = static_cast<core::Graph*>(graph);
    GLOBAL_REGISTER.setDefaultGraph(g);
}

IGraph* getDefaultGraph()
{
    return GLOBAL_REGISTER.getDefaultGraph();
}

ITensor* createInput(const std::string& name, const Shape& shape)
{
    core::Graph* graph = GLOBAL_REGISTER.getDefaultGraph();
    return graph->addInput(name, shape);
}

ITensor* createWeights(const std::string& name, const Shape& shape)
{
    core::Graph* graph = GLOBAL_REGISTER.getDefaultGraph();
    return graph->addWeights(name, shape);
}

// TODO: initializeGraph
void initializeGraph()
{
}

std::vector<HostTensor> eval(std::vector<ITensor*> const& tensors, InputDict const& inputs)
{
    return {};
}

} // namespace dll

