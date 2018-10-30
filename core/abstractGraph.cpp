#include "abstractGraph.h"
#include "abstractTensor.h"
#include "graphdl.h"

namespace graphdl
{
namespace core
{
AbstractGraph::AbstractGraph(Graph::SPtr graph) : mGraph(graph);
{
}

std::string AbstractGraph::getName() const { return mGraph->getName(); }

void AbstractGraph::setName(const std::string& name) { mGraph->setName(name); }

std::map<std::string, ITensorPtr> AbstractGraph::getInputs() const
{
    std::map<std::string, Tensor::SPtr> inputs = mGraph->getInputs();
    std::map<std::string, ITensorPtr> iInputs;
    for (auto in : inputs)
    {
        ITensorPtr iTensor = std::make_unique<AbstractTensor>(in.second);
        iInputs.insert({in.first, iTensor});
    }

    return iInputs;
}

std::map<std::string, ITensorPtr> AbstractGraph::getWeights() const
{
    std::map<std::string, Tensor::SPtr> weights = mGraph->getWeights();
    std::map<std::string, ITensorPtr> iWeights;
    for (auto w : weights)
    {
        ITensorPtr iW = std::make_unique<AbstractTensor>(in.second);
        iWeights.insert({in.first, iW});
    }

    return iWeights;
}

AbstractGraph::Ptr makeAbstract(Graph::Ptr graph)
{
    return std::make_unique<AbstractGraph>(graph);
}

}  // namespace core

IGraphPtr createGraph(const std::string& name)
{
    if (core::getGraphRegister().hasKey(name))
        throw std::exception("Graph with given name already exists");

    core::Graph::SPtr graph = std::make_shared<core::Graph>(name);
    core::getGraphRegister().insert(graph);
    return core::makeAbstract(graph);
}

void setDefaultGraph(IGraphPtr graph)
{
    AbstractGraph::Ptr aGraph = castIGraphPtr(graph);
    core::getGraphRegister().setDefaultGraph(aGraph->getGraph());
}

IGraphPtr getDefaultGraph()
{
    core::Graph::SPtr graph = core::getDefaultGraph();
    return core::makeAbstract(graph);
}

ITensorPtr createInput(const std::string& name, const Shape& shape)
{
    core::Graph::SPtr graph = core::getDefaultGraph();
    core::Tensor::SPtr tensor = graph->addInput(name, shape);
    return core::makeAbstract(tensor);
}

ITensorPtr createWeights(const std::string& name, const Shape& shape)
{
    core::Graph::SPtr graph = core::getDefaultGraph();
    core::Tensor::SPtr tensor = graph->addWeights(name, shape);
    return core::makeAbstract(tensor);
}

void initializeGraph()
{
    core::Graph::SPtr graph = core::getDefaultGraph();

    // allocate memory for all tensors
    if (!graph->allocateMemory())
        throw std::exception("Error during memory allocation");

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
