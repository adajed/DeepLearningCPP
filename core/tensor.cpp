#include "graph.h"
#include "layer.h"
#include "memory.h"

namespace graphdl
{
namespace core
{
Tensor::Tensor(Graph::SPtr graph, const std::string& name, const TensorShape& shape)
    : mID(graph->nextTensorID())
    , mName(name)
    , mShape(shape)
    , mIsEvaluated(false)
    , mLayer()
    , mGraph(graph)
    , mMemory(MemoryType::kHOST_MEMORY, shape.count())
{
}

Tensor::ID Tensor::getID() const { return mID; }

std::string Tensor::getName() const { return mName; }

void Tensor::setName(const std::string& name) { mName = name; }

TensorShape Tensor::getShape() const { return mShape; }

Layer::SPtr Tensor::getLayer() const { return mLayer.lock(); }

void Tensor::setLayer(Layer::SPtr layer) { mLayer = Layer::WeakPtr(layer); }

Graph::SPtr Tensor::getGraph() const { return mGraph.lock(); }

Memory Tensor::getMemory() { return mMemory; }

bool Tensor::allocateMemory() { return mMemory.allocate(); }

void Tensor::freeMemory() { mMemory.free(); }

void Tensor::eval(const InputDict& inputs)
{
    if (!mIsEvaluated)
    {
        mLayer.lock()->eval(inputs);
        mIsEvaluated = true;
    }
}

void Tensor::reset() { mIsEvaluated = false; }

Tensor::~Tensor() { mMemory.free(); }

Tensor::SPtr createTensor(const std::string& name, const TensorShape& shape)
{
    Graph::SPtr graph = core::getDefaultGraph();
    return std::make_shared<Tensor>(graph, name, shape);
}

}  // namespace core
}  // namespace graphdl
