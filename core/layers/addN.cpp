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
void runAddNHost(int n, std::size_t size, float** xs, float* y)
{
    for (std::size_t pos = 0; pos < size; ++pos)
    {
        y[pos] = 0.;
        for (int i = 0; i < n; ++i) y[pos] += xs[i][pos];
    }
}

void runAddNGradientHost(int n, std::size_t size, const float* yGrad,
                         float** xGrads)
{
    for (std::size_t pos = 0; pos < size; ++pos)
    {
        for (int i = 0; i < n; ++i) xGrads[i][pos] = yGrad[pos];
    }
}

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

AddNLayer::AddNLayer(ID id, std::vector<Tensor::SPtr> tensors)
    : DifferentiableLayer(
          id, tensors,
          {createTensor("", tensors[0]->getShape(), tensors[0]->getType())})
{
}

void AddNLayer::execute(const std::vector<float*>& inputs,
                        const std::vector<float*>& outputs,
                        const InputDict& /*inputDict*/)
{
    auto** xs = const_cast<float**>(inputs.data());
    float* y = outputs[0];
    size_t size = mOutputs[0]->getCount();

    if (mOutputs[0]->getType() == MemoryType::kHOST_MEMORY)
        runAddNHost(inputs.size(), size, xs, y);
#ifdef CUDA_AVAILABLE
    else
        cuda::runAddNDevice(inputs.size(), size, xs, y);
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

AddNGradientLayer::AddNGradientLayer(ID id,
                                     const std::vector<Tensor::SPtr>& ins,
                                     const Tensor::SPtr& out,
                                     const Tensor::SPtr& outGrad)
    : Layer(id, createGradientInputs(ins, out, outGrad),
            createGradientOutputs(ins))
{
    for (unsigned i = 1; i < ins.size(); ++i)
        assert(ins[0]->getType() == ins[i]->getType());
    assert(ins[0]->getType() == out->getType());
    assert(ins[0]->getType() == outGrad->getType());
}

void AddNGradientLayer::execute(const std::vector<float*>& inputs,
                                const std::vector<float*>& outputs,
                                const InputDict& /*inputDict*/)
{
    Tensor::SPtr t = getInputs().back();
    size_t size = t->getCount();
    float* yGrad = inputs.back();
    auto** xGrads = const_cast<float**>(outputs.data());

    if (t->getType() == MemoryType::kHOST_MEMORY)
        runAddNGradientHost(outputs.size(), size, yGrad, xGrads);
#ifdef CUDA_AVAILABLE
    else
        cuda::runAddNGradientDevice(outputs.size(), size, yGrad, xGrads);
#endif
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
