#include "abstractGraph.h"

#include "abstractTensor.h"
#include "graphdl.h"
#include "weightsNamespaces.h"

#include <cassert>
#include <utility>

namespace graphdl
{
namespace core
{
AbstractGraph::AbstractGraph(Graph::SPtr graph) : mGraph(std::move(graph)) {}

std::string AbstractGraph::getName() const
{
    return mGraph->getName();
}

void AbstractGraph::setName(const std::string& name)
{
    mGraph->setName(name);
}

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

Graph::SPtr AbstractGraph::get() const
{
    return mGraph;
}

AbstractGraph::Ptr makeAbstractGraph(Graph::SPtr graph)
{
    static std::map<Graph::SPtr, AbstractGraph::Ptr> sMap;

    if (sMap.count(graph) == 0)
        sMap[graph] = std::make_shared<AbstractGraph>(graph);

    return sMap[graph];
}

AbstractGraph::Ptr castIGraphPtr(const IGraphPtr& igraph)
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

void setDefaultGraph(const IGraphPtr& graph)
{
    core::AbstractGraph::Ptr aGraph = core::castIGraphPtr(graph);
    core::getGraphRegister().setDefaultGraph(aGraph->get());
}

IGraphPtr getDefaultGraph()
{
    core::Graph::SPtr graph = core::getDefaultGraph();
    return core::makeAbstractGraph(graph);
}

ITensorPtr createInput(const std::string& name, const Shape& shape,
                       MemoryLocation location)
{
#ifndef CUDA_AVAILABLE
    if (location == MemoryLocation::kDEVICE)
        throw std::runtime_error("Cuda not available, please use kHOST");
#endif
    auto inputs = core::getDefaultGraph()->getInputs();
    for (auto in : inputs)
        if (in.first == name)
            throw std::runtime_error("Input \"" + name + "\" already exists");

    core::MemoryType type = core::memoryLocationToType(location);
    core::Layer::SPtr input =
        core::createLayer<core::InputLayer>(name, shape, type);
    core::Tensor::SPtr tensor = core::getDefaultGraph()->addInput(name, input);
    return core::makeAbstractTensor(tensor);
}

ITensorPtr createWeights(const std::string& name, const Shape& shape,
                         MemoryLocation location)
{
#ifndef CUDA_AVAILABLE
    if (location == MemoryLocation::kDEVICE)
        throw std::runtime_error("Cuda not available, please use kHOST");
#endif
    auto graphWeights = core::getDefaultGraph()->getWeights();
    for (auto in : graphWeights)
        if (in.first == name)
            throw std::runtime_error("Weights \"" + name + "\" already exists");

    core::MemoryType type = core::memoryLocationToType(location);
    core::Layer::SPtr weights =
        core::createLayer<core::WeightsLayer>(name, shape, type);
    core::Tensor::SPtr tensor = core::getDefaultGraph()->addWeights(
        name, weights, core::GRAPH_WEIGHTS_NAMESPACE);
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

std::vector<HostTensor> eval(const std::vector<ITensorPtr>& tensors,
                             const InputDict& inputs)
{
    core::Graph::SPtr graph = core::getDefaultGraph();
    graph->prepareForNextComputation();

    std::vector<HostTensor> outputs;
    outputs.reserve(tensors.size());
    for (const auto& tensor : tensors) outputs.push_back(tensor->eval(inputs));
    return outputs;
}

}  // namespace graphdl
