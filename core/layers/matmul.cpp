#include "matmul.h"

#include "abstractTensor.h"
#include "graph.h"
#include "graphdl_ops.h"

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

void MatmulLayer::execute(const std::vector<float*>& inputs,
                          const std::vector<float*>& outputs,
                          const InputDict& /*inputDict*/)
{
    Tensor::SPtr tX1 = getInputs()[0];
    Tensor::SPtr tX2 = getInputs()[1];
    float* x1 = inputs[0];
    float* x2 = inputs[1];
    float* y = outputs[0];
    int n = tX1->getShape()[0];
    int m = tX2->getShape()[0];
    int k = tX2->getShape()[1];

    if (tX1->getType() == MemoryType::kHOST_MEMORY)
        runMatmulHost(n, m, k, x1, x2, y);
#ifdef CUDA_AVAILABLE
    else
        cuda::runMatmulDevice(n, m, k, x1, x2, y);
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

void MatmulGradientLayer::execute(const std::vector<float*>& inputs,
                                  const std::vector<float*>& outputs,
                                  const InputDict& /*inputDict*/)
{
    Tensor::SPtr tX1 = getInputs()[0];
    Tensor::SPtr tX2 = getInputs()[1];
    float* x1 = inputs[0];
    float* x2 = inputs[1];
    float* yGrad = inputs[3];
    float* x1Grad = outputs[0];
    float* x2Grad = outputs[1];

    size_t n = tX1->getShape()[0];
    size_t m = tX2->getShape()[0];
    size_t k = tX2->getShape()[1];

    if (tX1->getType() == MemoryType::kHOST_MEMORY)
        runMatmulGradientHost(n, m, k, x1, x2, yGrad, x1Grad, x2Grad);
#ifdef CUDA_AVAILABLE
    else
        cuda::runMatmulGradientDevice(n, m, k, x1, x2, yGrad, x1Grad, x2Grad);
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
