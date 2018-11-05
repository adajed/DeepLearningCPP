#include "addN.h"
#include <assert.h>
#include "abstractTensor.h"
#include "graph.h"
#include "graphdl_ops.h"

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

void runAddNGradientHost(int n, std::size_t size, float* yGrad, float** xGrads)
{
    for (std::size_t pos = 0; pos < size; ++pos)
    {
        for (int i = 0; i < n; ++i) xGrads[i][pos] = yGrad[pos];
    }
}

std::vector<Tensor::SPtr> createGradientInputs(std::vector<Tensor::SPtr> ins,
                                               Tensor::SPtr out,
                                               Tensor::SPtr outGrad)
{
    ins.push_back(out);
    ins.push_back(outGrad);
    return ins;
}

std::vector<Tensor::SPtr> createGradientOutputs(std::vector<Tensor::SPtr> ins)
{
    std::vector<Tensor::SPtr> outs;
    for (Tensor::SPtr i : ins)
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

void AddNLayer::execute(const InputDict& inputs)
{
    std::vector<Tensor::SPtr> ins = getInputs();
    std::vector<float*> xs;
    for (Tensor::SPtr in : ins)
    {
        in->eval(inputs);
        xs.push_back(in->getMemory().getValues());
    }
    float* output = mOutputs[0]->getMemory().getValues();
    std::size_t size = mOutputs[0]->getMemory().getCount();

    if (mOutputs[0]->getType() == MemoryType::kHOST_MEMORY)
        runAddNHost(xs.size(), size, xs.data(), output);
#ifdef CUDA_AVAILABLE
    else
        cuda::runAddNDevice(xs.size(), size, xs.data(), output);
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

AddNGradientLayer::AddNGradientLayer(ID id, std::vector<Tensor::SPtr> ins,
                                     Tensor::SPtr out, Tensor::SPtr outGrad)
    : Layer(id, createGradientInputs(ins, out, outGrad),
            createGradientOutputs(ins))
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

    std::size_t size = outputGrad->getMemory().getCount();
    float* yGrad = outputGrad->getMemory().getValues();
    std::vector<float*> xGrads;
    for (Tensor::SPtr xG : mOutputs)
        xGrads.push_back(xG->getMemory().getValues());

    if (outputGrad->getType() == MemoryType::kHOST_MEMORY)
        runAddNGradientHost(xGrads.size(), size, yGrad, xGrads.data());
#ifdef CUDA_AVAILABLE
    else
        cuda::runAddNGradientDevice(xGrads.size(), size, yGrad, xGrads.data());
#endif
}

}  // namespace layers

Tensor::SPtr addN(std::vector<Tensor::SPtr> tensors)
{
    if (tensors.size() == 0)
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
    for (ITensorPtr t : tensors)
        inputs.push_back(core::castITensorPtr(t)->get());

    return makeAbstractTensor(core::addN(inputs));
}

}  // namespace graphdl
