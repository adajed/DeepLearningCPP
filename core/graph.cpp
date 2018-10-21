#include <assert.h>

#include "dll.h"
#include "dll_errors.h"
#include "graph.h"

namespace dll
{
namespace core
{

const std::string GraphRegister::DEFAULT_GRAPH_NAME = "default_graph";

Graph::Graph(const std::string& name)
    : mName(name)
    , mInputOps()
    , mOps()
    , mTensors()
{}

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

bool Graph::allocateMemory()
{
    std::vector<Tensor*> allocatedTensors;
    for (std::pair<Tensor::ID, Tensor*> pair : mTensors)
    {
        if (!pair.second->allocateMemory())
        {
            for (Tensor* t : allocatedTensors)
                t->freeMemory();
            return false;
        }
        allocatedTensors.push_back(pair.second);
    }
    return true;
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
    mInputOps.insert({name, oper});
    addOper(oper);

    return oper->getOutputs()[0];
}

// TODO: Graph::addWeights
Tensor* Graph::addWeights(const std::string& name, const Shape& shape)
{
    return nullptr;
}

void Graph::addOper(Oper* oper)
{
    Oper::ID opID = oper->getID();
    mOps.insert({opID, oper});
    for (Tensor* tensor : oper->getOutputs())
    {
        Tensor::ID tensorID = tensor->getID();
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

    Graph* default_graph = new Graph(GraphRegister::DEFAULT_GRAPH_NAME);
    mGraphDict[GraphRegister::DEFAULT_GRAPH_NAME] = default_graph;
    mDefaultGraph = default_graph;
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

void initializeGraph()
{
    core::Graph* graph = GLOBAL_REGISTER.getDefaultGraph();
    if (!graph->allocateMemory())
        throw errors::MemoryAllocationError();
}

void eval(std::vector<ITensor*> const& tensors, InputDict const& inputs, std::vector<HostTensor*> hostTensors)
{
    assert(tensors.size() == hostTensors.size());

    core::Graph* graph = GLOBAL_REGISTER.getDefaultGraph();
    graph->reset();

    for (std::size_t i = 0; i < tensors.size(); ++i)
        tensors[i]->eval(inputs, hostTensors[i]);
}

} // namespace dll

