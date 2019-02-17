#include "abstractTensor.h"
#include "graph.h"
#include "graphdl_ops.h"
#include "matmul.h"

#include <cassert>

namespace graphdl
{
namespace core
{
namespace layers
{
namespace
{
std::vector<Tensor::SPtr> createOutput(const Tensor::SPtr& m1,
                                       const Tensor::SPtr& m2)
{
    assert(m1->getType() == m2->getType());
    return {createTensor("", {m1->getShape()[0], m2->getShape()[1]},
                         m1->getType())};
}

std::vector<Tensor::SPtr> createGradientOutputs(const Tensor::SPtr& m1,
                                                const Tensor::SPtr& m2)
{
    assert(m1->getType() == m2->getType());
    return {createTensor("", m1->getShape(), m1->getType()),
            createTensor("", m2->getShape(), m2->getType())};
}

}  // namespace

void runMatmulHost(const float* x1, const float* x2,
                   float* y, int n, int m, int k)
{
    for (int xPos = 0; xPos < n; ++xPos)
        for (int yPos = 0; yPos < k; ++yPos)
        {
            y[k * xPos + yPos] = 0.;
            for (int i = 0; i < m; ++i)
                y[k * xPos + yPos] += x1[m * xPos + i] * x2[k * i + yPos];
        }
}

void runMatmulGradientHost(const float* x1, const float* x2,
                           const float* yGrad, float* x1Grad,
                           float* x2Grad, int n, int m, int k)
{
    // calculate x1Grad
    for (int xPos = 0; xPos < n; ++xPos)
    {
        for (int yPos = 0; yPos < m; ++yPos)
        {
            x1Grad[m * xPos + yPos] = 0.;
            for (int i = 0; i < k; ++i)
                x1Grad[m * xPos + yPos] += x2[k * yPos + i] * yGrad[k * xPos + i];
        }
    }

    // calculate x2Grad
    for (int xPos = 0; xPos < m; ++xPos)
    {
        for (int yPos = 0; yPos < k; ++yPos)
        {
            x2Grad[k * xPos + yPos] = 0.;
            for (int i = 0; i < n; ++i)
                x2Grad[k * xPos + yPos] += x1[m * i + xPos] * yGrad[k * i + yPos];
        }
    }
}

MatmulLayer::MatmulLayer(ID id, const Tensor::SPtr& m1, const Tensor::SPtr& m2)
    : DifferentiableLayer(id, {m1, m2}, createOutput(m1, m2))
{
    assert(m1->getShape().size() == 2);
    assert(m2->getShape().size() == 2);
    assert(m1->getShape()[1] == m2->getShape()[0]);
}

void MatmulLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr x1Tensor = mInputs[0].lock();
    Tensor::SPtr x2Tensor = mInputs[1].lock();
    x1Tensor->eval(inputs);
    x2Tensor->eval(inputs);

    float* x1 = x1Tensor->getMemory().getValues();
    float* x2 = x2Tensor->getMemory().getValues();
    float* y = mOutputs[0]->getMemory().getValues();
    int n = x1Tensor->getShape()[0];
    int m = x2Tensor->getShape()[0];
    int k = x2Tensor->getShape()[1];

    if (mOutputs[0]->getType() == MemoryType::kHOST_MEMORY)
        runMatmulHost(x1, x2, y, n, m, k);
#ifdef CUDA_AVAILABLE
    else
        cuda::runMatmulDevice(x1, x2, y, n, m, k);
#endif
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

MatmulGradientLayer::MatmulGradientLayer(ID id, const Tensor::SPtr& m1,
                                         const Tensor::SPtr& m2,
                                         const Tensor::SPtr& out,
                                         const Tensor::SPtr& outGrad)
    : Layer(id, {m1, m2, out, outGrad}, createGradientOutputs(m1, m2))
{
    assert(m1->getShape().size() == 2);
    assert(m2->getShape().size() == 2);
    assert(m1->getShape()[1] == m2->getShape()[0]);
    assert(out->getShape()[0] == m1->getShape()[0] &&
           out->getShape()[1] == m2->getShape()[1]);
    assert(out->getShape() == outGrad->getShape());
}

void MatmulGradientLayer::execute(const InputDict& inputs)
{
    Tensor::SPtr x1Tensor = mInputs[0].lock();
    Tensor::SPtr x2Tensor = mInputs[1].lock();
    Tensor::SPtr yGradTensor = mInputs[3].lock();
    x1Tensor->eval(inputs);
    x2Tensor->eval(inputs);
    yGradTensor->eval(inputs);

    float* x1 = x1Tensor->getMemory().getValues();
    float* x2 = x2Tensor->getMemory().getValues();
    float* yGrad = yGradTensor->getMemory().getValues();
    float* x1Grad = mOutputs[0]->getMemory().getValues();
    float* x2Grad = mOutputs[1]->getMemory().getValues();

    int n = x1Tensor->getShape()[0];
    int m = x1Tensor->getShape()[1];
    int k = x2Tensor->getShape()[1];

    if (x1Tensor->getType() == MemoryType::kHOST_MEMORY)
        runMatmulGradientHost(x1, x2, yGrad, x1Grad, x2Grad, n, m, k);
#ifdef CUDA_AVAILABLE
    else
        cuda::runMatmulGradientDevice(x1, x2, yGrad, x1Grad, x2Grad, n, m, k);
#endif
}

}  // namespace layers

Tensor::SPtr matmul(const Tensor::SPtr& m1, const Tensor::SPtr& m2)
{
    if (m1->getShape().size() != 2 || m2->getShape().size() != 2)
        throw std::runtime_error("Shapes don\'t match");
    if (m1->getShape()[1] != m2->getShape()[0])
        throw std::runtime_error("Shapes don\'t match");

    Layer::SPtr layer = createLayer<layers::MatmulLayer>(m1, m2);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr matmul(const ITensorPtr& m1, const ITensorPtr& m2)
{
    core::AbstractTensor::Ptr mat1 = core::castITensorPtr(m1);
    core::AbstractTensor::Ptr mat2 = core::castITensorPtr(m2);
    return makeAbstractTensor(core::matmul(mat1->get(), mat2->get()));
}

}  // namespace graphdl
