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
void runMatmulHost(int n, int m, int k, const float* X1, const float* X2,
                   float* Y)
{
    for (int x = 0; x < n; ++x)
        for (int y = 0; y < k; ++y)
        {
            Y[k * x + y] = 0.;
            for (int i = 0; i < m; ++i)
                Y[k * x + y] += X1[m * x + i] * X2[k * i + y];
        }
}

void runMatmulGradientHost(int n, int m, int k, const float* X1,
                           const float* X2, const float* Ygrad, float* X1grad,
                           float* X2grad)
{
    for (int x = 0; x < n; ++x)
        for (int y = 0; y < m; ++y)
        {
            X1grad[m * x + y] = 0.;
            for (int i = 0; i < k; ++i)
                X1grad[m * x + y] += X2[k * y + i] * Ygrad[k * x + i];
        }
    for (int x = 0; x < m; ++x)
        for (int y = 0; y < k; ++y)
        {
            X2grad[k * x + y] = 0.;
            for (int i = 0; i < n; ++i)
                X2grad[k * x + y] += X1[m * i + x] * Ygrad[k * i + y];
        }
}

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

MatmulLayer::MatmulLayer(ID id, const Tensor::SPtr& m1, const Tensor::SPtr& m2)
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

    float* in1 = m1->getMemory().getValues();
    float* in2 = m2->getMemory().getValues();
    float* out = mOutputs[0]->getMemory().getValues();
    int n = m1->getShape()[0];
    int m = m2->getShape()[0];
    int k = m2->getShape()[1];

    if (mOutputs[0]->getType() == MemoryType::kHOST_MEMORY)
        runMatmulHost(n, m, k, in1, in2, out);
#ifdef CUDA_AVAILABLE
    else
        cuda::runMatmulDevice(n, m, k, in1, in2, out);
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
    Tensor::SPtr m1 = mInputs[0].lock();
    Tensor::SPtr m2 = mInputs[1].lock();
    Tensor::SPtr outGrad = mInputs[3].lock();
    m1->eval(inputs);
    m2->eval(inputs);
    outGrad->eval(inputs);

    float* in1 = m1->getMemory().getValues();
    float* in2 = m2->getMemory().getValues();
    float* outG = outGrad->getMemory().getValues();
    float* grad1 = mOutputs[0]->getMemory().getValues();
    float* grad2 = mOutputs[1]->getMemory().getValues();

    std::size_t n = m1->getShape()[0];
    std::size_t m = m1->getShape()[1];
    std::size_t k = m2->getShape()[1];

    if (m1->getType() == MemoryType::kHOST_MEMORY)
        runMatmulGradientHost(n, m, k, in1, in2, outG, grad1, grad2);
#ifdef CUDA_AVAILABLE
    else
        cuda::runMatmulGradientDevice(n, m, k, in1, in2, outG, grad1, grad2);
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
