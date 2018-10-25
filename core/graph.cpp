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
    , mWeightsOps()
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


std::map<std::string, ITensorSPtr> Graph::getInputs() const
{
    std::map<std::string, ITensorSPtr> inputs;
    for (auto pair : mInputOps)
        inputs.insert({pair.first, ITensorSPtr(pair.second->getOutputs()[0])});
    return inputs;
}

std::map<std::string, ITensorSPtr> Graph::getWeights() const
{
    std::map<std::string, ITensorSPtr> weights;
    for (auto pair : mWeightsOps)
        weights.insert({pair.first, ITensorSPtr(pair.second->getOutputs()[0])});
    return weights;
}

void Graph::reset()
{
    for (auto pair : mInputOps)
        pair.second->reset();
    for (auto pair : mWeightsOps)
        pair.second->reset();
}

bool Graph::allocateMemory()
{
    std::vector<Tensor::SPtr> allocatedTensors;
    for (std::pair<Tensor::ID, Tensor::SPtr> pair : mTensors)
    {
        if (!pair.second->allocateMemory())
        {
            for (Tensor::SPtr t : allocatedTensors)
                t->freeMemory();
            return false;
        }
        allocatedTensors.push_back(pair.second);
    }
    return true;
}

void Graph::initializeOpers()
{
    for (auto pair : mOps)
        pair.second->initialize();
}

void Graph::freeMemory()
{
    for (auto pair : mTensors)
        pair.second->freeMemory();
}


Tensor::SPtr Graph::addInput(const std::string& name, const Shape& shape)
{
    if (mInputOps.count(name) > 0)
        return nullptr;

    std::shared_ptr<InputOper> oper = std::make_shared<InputOper>(name, shape);
    mInputOps.insert({name, oper});
    insertOperation(Oper::SPtr(oper));

    return oper->getOutputs()[0];
}

Tensor::SPtr Graph::addWeights(const std::string& name, const Shape& shape)
{
    if (mWeightsOps.count(name) > 0)
        return nullptr;

    auto weightsOper = std::make_shared<WeightsOper>(name, shape);
    mWeightsOps.insert({name, weightsOper});
    insertOperation(Oper::SPtr(weightsOper));

    return weightsOper->getOutputs()[0];
}

void Graph::insertOperation(Oper::SPtr oper)
{
    Oper::ID opID = oper->getID();
    mOps.insert({opID, oper});
    for (Tensor::SPtr tensor : oper->getOutputs())
    {
        Tensor::ID tensorID = tensor->getID();
        tensor->setOper(oper);
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
    if (hasKey(graph->getName()))
        return false;
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
    Graph::SPtr default_graph = std::make_shared<Graph>(GraphRegister::DEFAULT_GRAPH_NAME);
    mGraphDict[GraphRegister::DEFAULT_GRAPH_NAME] = default_graph;
    mDefaultGraph = default_graph;
}

Graph::SPtr getDefaultGraph()
{
    return GraphRegister::getGlobalGraphRegister().getDefaultGraph();
}

} // namespace core

#define GLOBAL_REGISTER core::GraphRegister::getGlobalGraphRegister()

IGraphSPtr createGraph(const std::string& name)
{
    if (GLOBAL_REGISTER.hasKey(name))
        return nullptr;

    core::Graph::SPtr graph = std::make_shared<core::Graph>(name);
    GLOBAL_REGISTER.insert(graph);
    return IGraphSPtr(graph);
}

void setDefaultGraph(IGraphSPtr graph)
{
    core::Graph::SPtr g = std::static_pointer_cast<core::Graph>(graph);
    GLOBAL_REGISTER.setDefaultGraph(g);
}

IGraphSPtr getDefaultGraph()
{
    return IGraphSPtr(GLOBAL_REGISTER.getDefaultGraph());
}

ITensorSPtr createInput(const std::string& name, const Shape& shape)
{
    core::Graph::SPtr graph = core::getDefaultGraph();
    return ITensorSPtr(graph->addInput(name, shape));
}

ITensorSPtr createWeights(const std::string& name, const Shape& shape)
{
    core::Graph::SPtr graph = core::getDefaultGraph();
    return ITensorSPtr(graph->addWeights(name, shape));
}

void initializeGraph()
{
    core::Graph::SPtr graph = core::getDefaultGraph();
    // allocate memory for all tensors
    if (!graph->allocateMemory())
        throw errors::MemoryAllocationError();

    // initialize all the operations
    graph->initializeOpers();
}

void eval(std::vector<ITensorSPtr> const& tensors, InputDict const& inputs, std::vector<HostTensor> hostTensors)
{
    assert(tensors.size() == hostTensors.size());

    core::Graph::SPtr graph = core::getDefaultGraph();
    graph->reset();

    for (std::size_t i = 0; i < tensors.size(); ++i)
        tensors[i]->eval(inputs, hostTensors[i]);
}

} // namespace dll

