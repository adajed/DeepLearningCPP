#include "abstractGraph.h"
#include <assert.h>
#include "abstractTensor.h"
#include "graphdl.h"

namespace graphdl
{
namespace core
{
AbstractGraph::AbstractGraph(Graph::SPtr graph) : mGraph(graph) {}

std::string AbstractGraph::getName() const { return mGraph->getName(); }

void AbstractGraph::setName(const std::string& name) { mGraph->setName(name); }

std::map<std::string, ITensorPtr> AbstractGraph::getInputs() const
{
    std::map<std::string, Tensor::SPtr> inputs = mGraph->getInputs();
    std::map<std::string, ITensorPtr> iInputs;
    for (auto in : inputs)
        iInputs[in.first] = std::make_unique<AbstractTensor>(in.second);

    return iInputs;
}

std::map<std::string, ITensorPtr> AbstractGraph::getWeights() const
{
    std::map<std::string, Tensor::SPtr> weights = mGraph->getWeights();
    std::map<std::string, ITensorPtr> iWeights;
    for (auto w : weights)
        iWeights[w.first] = std::make_unique<AbstractTensor>(w.second);

    return iWeights;
}

Graph::SPtr AbstractGraph::get() const { return mGraph; }

AbstractGraph::Ptr makeAbstractGraph(Graph::SPtr graph)
{
    static std::map<Graph::SPtr, AbstractGraph::Ptr> sMap;

    if (sMap.count(graph) == 0)
        sMap[graph] = std::make_shared<AbstractGraph>(graph);

    return sMap[graph];
}

AbstractGraph::Ptr castIGraphPtr(IGraphPtr igraph)
{
    return std::static_pointer_cast<AbstractGraph>(igraph);
}

}  // namespace core

IGraphPtr createGraph(const std::string& name)
{
    if (core::getGraphRegister().hasKey(name))
        throw std::runtime_error("Graph with given name already exists");

    core::Graph::SPtr graph = std::make_shared<core::Graph>(name);
    core::getGraphRegister().insert(graph);
    return core::makeAbstractGraph(graph);
}

void setDefaultGraph(IGraphPtr graph)
{
    core::AbstractGraph::Ptr aGraph = core::castIGraphPtr(graph);
    core::getGraphRegister().setDefaultGraph(aGraph->get());
}

IGraphPtr getDefaultGraph()
{
    core::Graph::SPtr graph = core::getDefaultGraph();
    return core::makeAbstractGraph(graph);
}

ITensorPtr createInput(const std::string& name, const Shape& shape)
{
    core::Layer::SPtr input = core::createLayer<core::InputLayer>(name, shape);
    core::Tensor::SPtr tensor = core::getDefaultGraph()->addInput(name, input);
    return core::makeAbstractTensor(tensor);
}

ITensorPtr createWeights(const std::string& name, const Shape& shape)
{
    core::Layer::SPtr weights =
        core::createLayer<core::WeightsLayer>(name, shape);
    core::Tensor::SPtr tensor =
        core::getDefaultGraph()->addWeights(name, weights);
    return core::makeAbstractTensor(tensor);
}

void initializeGraph()
{
    core::Graph::SPtr graph = core::getDefaultGraph();

    // allocate memory for all tensors
    if (!graph->allocateMemory())
        throw std::runtime_error("Error during memory allocation");

    // initialize all the layers
    graph->initializeLayers();
}

void eval(const std::vector<ITensorPtr>& tensors, const InputDict& inputs,
          const std::vector<HostTensor>& hostTensors)
{
    assert(tensors.size() == hostTensors.size());

    core::Graph::SPtr graph = core::getDefaultGraph();
    graph->prepareForNextComputation();

    for (std::size_t i = 0; i < tensors.size(); ++i)
        tensors[i]->eval(inputs, hostTensors[i]);
}

}  // namespace graphdl
