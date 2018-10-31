#include <assert.h>
#include "abstractTensor.h"
#include "graph.h"
#include "graphdl_ops.h"
#include "matmul.h"

namespace graphdl
{
namespace core
{
namespace layers
{
std::vector<Tensor::SPtr> createOutput(Tensor::SPtr m1, Tensor::SPtr m2)
{
    return {createTensor("", {m1->getShape()[0], m2->getShape()[1]})};
}

MatmulLayer::MatmulLayer(ID id, Tensor::SPtr m1, Tensor::SPtr m2)
    : DifferentiableLayer(id, {m1, m2}, createOutput(m1, m2))
{
    assert(m1->getShape().size() == 2);
    assert(m2->getShape().size() == 2);
    assert(m1->getShape()[1] == m2->getShape()[0]);
}

void MatmulLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr m1 = mInputs[0].lock();
    Tensor::SPtr m2 = mInputs[1].lock();
    m1->eval(inputs);
    m2->eval(inputs);

    Memory in1 = m1->getMemory();
    Memory in2 = m2->getMemory();
    Memory out = mOutputs[0]->getMemory();

    int m = m2->getShape()[0];
    int k = mOutputs[0]->getShape()[1];

    for (std::size_t pos = 0; pos < out.getCount(); ++pos)
    {
        int x = pos / k;
        int y = pos % k;

        out[pos] = 0.;
        for (int i = 0; i < m; ++i) out[pos] += in1[m * x + i] * in2[k * i + y];
    }
}

Layer::TensorMap MatmulLayer::gradients(Tensor::SPtr output,
                                        Tensor::SPtr outputGrad)
{
    assert(output == mOutputs[0]);

    std::vector<Tensor::SPtr> inputs = getInputs();
    Layer::SPtr layer = createLayer<MatmulGradientLayer>(inputs[0], inputs[1],
                                                         output, outputGrad);

    std::vector<Tensor::SPtr> grads = layer->getOutputs();
    return {{inputs[0], grads[0]}, {inputs[1], grads[1]}};
}

std::vector<Tensor::SPtr> createGradientOutputs(Tensor::SPtr m1,
                                                Tensor::SPtr m2)
{
    return {createTensor("", m1->getShape()), createTensor("", m2->getShape())};
}

MatmulGradientLayer::MatmulGradientLayer(ID id, Tensor::SPtr m1,
                                         Tensor::SPtr m2, Tensor::SPtr out,
                                         Tensor::SPtr outGrad)
    : Layer(id, {m1, m2, out, outGrad}, createGradientOutputs(m1, m2))
{
    assert(m1->getShape().size() == 2);
    assert(m2->getShape().size() == 2);
    assert(m1->getShape()[1] == m2->getShape()[0]);
    assert(out->getShape()[0] =
               m1->getShape()[0] && out->getShape()[1] == m2->getShape()[1]);
    assert(out->getShape() == outGrad->getShape());
}

void MatmulGradientLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr m1 = mInputs[0].lock();
    Tensor::SPtr m2 = mInputs[1].lock();
    Tensor::SPtr outGrad = mInputs[3].lock();
    m1->eval(inputs);
    m2->eval(inputs);
    outGrad->eval(inputs);

    Memory in1 = m1->getMemory();
    Memory in2 = m2->getMemory();
    Memory outG = outGrad->getMemory();
    Memory grad1 = mOutputs[0]->getMemory();
    Memory grad2 = mOutputs[1]->getMemory();

    int n = m1->getShape()[0];
    int m = m1->getShape()[1];
    int k = m2->getShape()[1];

    for (std::size_t pos = 0; pos < grad1.getCount(); ++pos)
    {
        int x = pos / m;
        int y = pos % m;

        grad1[pos] = 0.;
        for (int i = 0; i < k; ++i)
            grad1[pos] += in2[k * y + i] * outG[k * x + i];
    }
    for (std::size_t pos = 0; pos < grad2.getCount(); ++pos)
    {
        int x = pos / k;
        int y = pos % k;

        grad2[pos] = 0.;
        for (int i = 0; i < n; ++i)
            grad2[pos] += in1[m * i + x] * outG[k * i + y];
    }
}

}  // namespace layers

Tensor::SPtr matmul(Tensor::SPtr m1, Tensor::SPtr m2)
{
    if (m1->getShape().size() != 2 || m2->getShape().size() != 2)
        throw std::runtime_error("Shapes don\'t match");
    else if (m1->getShape()[1] != m2->getShape()[0])
        throw std::runtime_error("Shapes don\'t match");

    Layer::SPtr layer = createLayer<layers::MatmulLayer>(m1, m2);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr matmul(ITensorPtr m1, ITensorPtr m2)
{
    core::AbstractTensor::Ptr mat1 = core::castITensorPtr(m1);
    core::AbstractTensor::Ptr mat2 = core::castITensorPtr(m2);
    return makeAbstractTensor(core::matmul(mat1->get(), mat2->get()));
}

}  // namespace graphdl
