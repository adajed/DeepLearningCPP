#include "layer.h"

#include "graph.h"

#include <utility>

namespace graphdl
{
namespace core
{
Layer::Layer(ID id, const std::vector<Tensor::SPtr>& inputs,
             std::vector<Tensor::SPtr> outputs)
    : mID(id), mIsEvaluated(false), mOutputs(std::move(outputs))
{
    for (const Tensor::SPtr& in : inputs)
        mInputs.push_back(Tensor::WeakPtr(in));
}

Layer::ID Layer::getID() const
{
    return mID;
}

Graph::SPtr Layer::getGraph() const
{
    return mGraph.lock();
}

void Layer::setGraph(const Graph::SPtr& graph)
{
    mGraph = Graph::WeakPtr(graph);
}

std::vector<Tensor::SPtr> Layer::getInputs()
{
    std::vector<Tensor::SPtr> inputs;
    for (const Tensor::WeakPtr& input : mInputs) inputs.push_back(input.lock());
    return inputs;
}

std::vector<Tensor::SPtr> Layer::getOutputs()
{
    return mOutputs;
}

std::set<Tensor::SPtr> Layer::getNecessaryInputs() const
{
    std::set<Tensor::SPtr> inputs;
    for (const Tensor::WeakPtr& tensor : mInputs)
    {
        auto tensorInputs = tensor.lock()->getNecessaryInputs();
        inputs.insert(tensorInputs.begin(), tensorInputs.end());
    }

    return inputs;
}

void Layer::eval(const InputDict& inputDict)
{
    if (!mIsEvaluated)
    {
        // calculate inputs
        std::vector<float*> inputs;
        for (const auto& in : mInputs)
        {
            Tensor::SPtr tensor = in.lock();
            tensor->eval(inputDict);
            inputs.push_back(tensor->getMemory().getValues());
        }

        std::vector<float*> outputs;
        for (const auto& out : mOutputs)
            outputs.push_back(out->getMemory().getValues());

        // calculate actual operation
        execute(inputs, outputs, inputDict);
        mIsEvaluated = true;
    }
}

void Layer::reset()
{
    mIsEvaluated = false;
    for (const Tensor::SPtr& output : mOutputs) output->reset();
}

Layer::~Layer() = default;

}  // namespace core
}  // namespace graphdl
