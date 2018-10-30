#include "graph.h"
#include "layer.h"

namespace graphdl
{
namespace core
{
Layer::Layer(Graph::SPtr graph, const std::vector<Tensor::SPtr>& inputs,
             const std::vector<Tensor::SPtr>& outputs)
    : mID(graph->nextLayerID())
    , mIsEvaluated(false)
    , mGraph(graph)
    , mInputs()
    , mOutputs(outputs)
{
    for (Tensor::SPtr in : inputs)
        mInputs.push_back(Tensor::WeakPtr(in));
}

Layer::ID Layer::getID() const { return mID; }

Graph::SPtr Layer::getGraph() const { return mGraph.lock(); }

std::vector<Tensor::SPtr> Layer::getInputs()
{
    std::vector<Tensor::SPtr> inputs;
    for (Tensor::WeakPtr input : mInputs) inputs.push_back(input.lock());
    return inputs;
}

std::vector<Tensor::SPtr> Layer::getOutputs() { return mOutputs; }

void Layer::eval(const InputDict& inputs)
{
    if (!mIsEvaluated)
    {
        // calculate actual operation
        executeOper(inputs);
        mIsEvaluated = true;
    }
}

void Layer::reset()
{
    mIsEvaluated = false;
    for (Tensor::SPtr output : mOutputs) output->reset();
}

}  // namespace core
}  // namespace graphdl
