#include <utility>

#include "graph.h"
#include "layer.h"
#include "memory.h"

namespace graphdl
{
namespace core
{
Tensor::Tensor(ID id, std::string name, const TensorShape& shape)
    : mID(id),
      mName(std::move(name)),
      mShape(shape),
      mIsEvaluated(false),

      mMemory(MemoryType::kHOST_MEMORY, shape.getCount())
{
}

Tensor::ID Tensor::getID() const { return mID; }

std::string Tensor::getName() const { return mName; }

void Tensor::setName(const std::string& name) { mName = name; }

TensorShape Tensor::getShape() const { return mShape; }

Layer::SPtr Tensor::getLayer() const { return mLayer.lock(); }

void Tensor::setLayer(const Layer::SPtr& layer)
{
    mLayer = Layer::WeakPtr(layer);
}

Graph::SPtr Tensor::getGraph() const { return mLayer.lock()->getGraph(); }

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
    return std::make_shared<Tensor>(graph->nextTensorID(), name, shape);
}

}  // namespace core
}  // namespace graphdl
