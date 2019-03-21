#include "addN.h"

#include "abstractTensor.h"
#include "graph.h"
#include "graphdl_ops.h"

#include <cassert>
#include <utility>

namespace graphdl
{
namespace core
{
namespace layers
{
namespace
{
std::vector<Tensor::SPtr> createGradientInputs(std::vector<Tensor::SPtr> ins,
                                               const Tensor::SPtr& out,
                                               const Tensor::SPtr& outGrad)
{
    ins.push_back(out);
    ins.push_back(outGrad);
    return ins;
}

std::vector<Tensor::SPtr> createGradientOutputs(std::vector<Tensor::SPtr> ins)
{
    std::vector<Tensor::SPtr> outs;
    outs.reserve(ins.size());
    for (const Tensor::SPtr& i : ins)
        outs.push_back(createTensor("", i->getShape(), i->getType()));
    return outs;
}

}  // namespace

void runAddNHost(float** xs, int n, float* y, size_t size)
{
    for (size_t pos = 0; pos < size; ++pos)
    {
        y[pos] = 0.;
        for (int i = 0; i < n; ++i) y[pos] += xs[i][pos];
    }
}

void runAddNGradientHost(const float* yGrad, float** xGrads, int n, size_t size)
{
    for (size_t pos = 0; pos < size; ++pos)
    {
        for (int i = 0; i < n; ++i) xGrads[i][pos] = yGrad[pos];
    }
}

AddNLayer::AddNLayer(ID id, std::vector<Tensor::SPtr> tensors)
    : DifferentiableLayer(
          id, tensors,
          {createTensor("", tensors[0]->getShape(), tensors[0]->getType())}),
      mArray(tensors[0]->getType(), tensors.size())
{
}

void AddNLayer::execute(const InputDict& inputs)
{
    std::vector<Tensor::SPtr> xsTensor = getInputs();
    for (const Tensor::SPtr& xTensor : xsTensor) xTensor->eval(inputs);

    float* y = mOutputs[0]->getMemory().getValues();
    size_t size = mOutputs[0]->getMemory().getCount();

    if (mOutputs[0]->getType() == MemoryType::kHOST_MEMORY)
        runAddNHost(mArray.getValues(), mArray.getCount(), y, size);
#ifdef CUDA_AVAILABLE
    else
        cuda::runAddNDevice(mArray.getValues(), mArray.getCount(), y, size);
#endif
}

Layer::TensorMap AddNLayer::gradients(Tensor::SPtr out, Tensor::SPtr outGrad)
{
    assert(out == mOutputs[0]);

    std::vector<Tensor::SPtr> inputs = getInputs();
    Layer::SPtr layer = createLayer<AddNGradientLayer>(inputs, out, outGrad);

    std::vector<Tensor::SPtr> grads = layer->getOutputs();
    Layer::TensorMap gradMap;
    for (unsigned i = 0; i < inputs.size(); ++i)
        gradMap.insert({inputs[i], grads[i]});
    return gradMap;
}

void AddNLayer::initialize()
{
    mArray.allocate();

    std::vector<Tensor::SPtr> tensors = getInputs();
    std::vector<float*> arr;
    arr.reserve(tensors.size());
    for (auto& t : tensors) arr.push_back(t->getMemory().getValues());

    mArray.fillFrom(arr.data());
}

AddNLayer::~AddNLayer()
{
    mArray.free();
}

AddNGradientLayer::AddNGradientLayer(ID id,
                                     const std::vector<Tensor::SPtr>& ins,
                                     const Tensor::SPtr& out,
                                     const Tensor::SPtr& outGrad)
    : Layer(id, createGradientInputs(ins, out, outGrad),
            createGradientOutputs(ins)),
      mArray(ins[0]->getType(), ins.size())
{
    for (unsigned i = 1; i < ins.size(); ++i)
        assert(ins[0]->getType() == ins[i]->getType());
    assert(ins[0]->getType() == out->getType());
    assert(ins[0]->getType() == outGrad->getType());
}

void AddNGradientLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr outputGrad = mInputs.back().lock();
    outputGrad->eval(inputs);

    size_t size = outputGrad->getMemory().getCount();
    float* yGrad = outputGrad->getMemory().getValues();

    if (outputGrad->getType() == MemoryType::kHOST_MEMORY)
        runAddNGradientHost(yGrad, mArray.getValues(), mArray.getCount(), size);
#ifdef CUDA_AVAILABLE
    else
        cuda::runAddNGradientDevice(yGrad, mArray.getValues(),
                                    mArray.getCount(), size);
#endif
}

void AddNGradientLayer::initialize()
{
    mArray.allocate();

    std::vector<float*> arr;
    for (auto& t : mOutputs) arr.push_back(t->getMemory().getValues());

    mArray.fillFrom(arr.data());
}

AddNGradientLayer::~AddNGradientLayer()
{
    mArray.free();
}

}  // namespace layers

Tensor::SPtr addN(std::vector<Tensor::SPtr> tensors)
{
    if (tensors.empty())
        throw std::runtime_error("List of input tensors cannot be empty");
    for (unsigned i = 1; i < tensors.size(); ++i)
    {
        if (tensors[0]->getShape() != tensors[i]->getShape())
            throw std::runtime_error("Shapes of inputs tensors don\'t match");
        if (tensors[0]->getType() != tensors[i]->getType())
            throw std::runtime_error(
                "Input tensors must be on the same device type");
    }

    Layer::SPtr layer = createLayer<layers::AddNLayer>(tensors);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr addN(std::vector<ITensorPtr> tensors)
{
    std::vector<core::Tensor::SPtr> inputs;
    inputs.reserve(tensors.size());
    for (const ITensorPtr& t : tensors)
        inputs.push_back(core::castITensorPtr(t)->get());

    return makeAbstractTensor(core::addN(inputs));
}

}  // namespace graphdl
