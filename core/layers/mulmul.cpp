#include <assert.h>
#include "dll_errors.h"
#include "dll_ops.h"
#include "graph.h"
#include "matmul.h"

namespace dll
{
namespace core
{
namespace layers
{
std::vector<Tensor::SPtr> createOutputGrad(
        Tensor::SPtr m1, Tensor::SPtr m2)
{
    return {createTensor("", m1->shape()), createTensor("", m2->shape())};
}

MatmulGradientOper::MatmulGradientOper(Tensor::SPtr m1, Tensor::SPtr m2,
                   Tensor::SPtr out, Tensor::SPtr outGrad)
    : Oper({m1, m2, out, outGrad}, createOutputGrad(m1, m2))
{
    assert(m1->shape().size() == 2);
    assert(m2->shape().size() == 2);
    assert(m1->shape()[1] == m2->shape()[0]);
    assert(out->shape()[0] = m1->shape()[0] &&
           out->shape()[1] == m2->shape()[1]);
    assert(out->shape() == outGrad->shape());
}


void MatmulGradientOper::executeOper(const InputDict& inputs)
{
    Tensor::SPtr m1 = mInputs[0].lock();
    Tensor::SPtr m2 = mInputs[1].lock();
    Tensor::SPtr outGrad = mInputs[3].lock();
    m1->exec(inputs);
    m2->exec(inputs);
    outGrad->exec(inputs);

    Memory in1 = m1->getMemory();
    Memory in2 = m2->getMemory();
    Memory outG = outGrad->getMemory();
    Memory grad1 = mOutputs[0]->getMemory();
    Memory grad2 = mOutputs[1]->getMemory();

    int n = m1->shape()[0];
    int m = m1->shape()[1];
    int k = m2->shape()[1];

    for (std::size_t pos = 0; pos < grad1.count(); ++pos)
    {
        int x = pos / m;
        int y = pos % m;

        grad1[pos] = 0.;
        for (int i = 0; i < k; ++i)
            grad1[pos] += outG[k * x + i] * in2[m * y + i];
    }
    for (std::size_t pos = 0; pos < grad2.count(); ++pos)
    {
        int x = pos / k;
        int y = pos % k;

        grad2[pos] = 0.;
        for (int i = 0; i < n; ++i)
            grad2[pos] += in1[m * i + x] * outG[k * i + y];
    }
}

std::vector<Tensor::SPtr> createOutput(
        Tensor::SPtr m1, Tensor::SPtr m2)
{
    TensorShape newShape({m1->shape()[0], m2->shape()[1]});
    return {createTensor("", newShape)};
}

MatmulOper::MatmulOper(Tensor::SPtr m1, Tensor::SPtr m2)
    : GradientOper({m1, m2}, createOutput(m1, m2))
{
    assert(m1->shape().size() == 2);
    assert(m2->shape().size() == 2);
    assert(m1->shape()[1] == m2->shape()[0]);
}

void MatmulOper::executeOper(const InputDict& inputs)
{
    Tensor::SPtr m1 = mInputs[0].lock();
    Tensor::SPtr m2 = mInputs[1].lock();
    m1->exec(inputs);
    m2->exec(inputs);

    Memory in1 = m1->getMemory();
    Memory in2 = m2->getMemory();
    Memory out = mOutputs[0]->getMemory();

    int m = m2->shape()[0];
    int k = mOutputs[0]->shape()[1];

    for (std::size_t pos = 0; pos < out.count(); ++pos)
    {
        int x = pos / k;
        int y = pos % k;

        out[pos] = 0.;
        for (int i = 0; i < m; ++i)
            out[pos] += in1[m * x + i] * in2[k * i + y];
    }
}

GradientOper::TensorMap MatmulOper::gradients(
        Tensor::SPtr output, Tensor::SPtr outputGrad)
{
    assert(output == mOutputs[0]);

    std::vector<Tensor::SPtr> inputs = getInputs();
    Oper::SPtr oper = std::make_shared<MatmulGradientOper>(
            inputs[0], inputs[1], output, outputGrad);
    getDefaultGraph()->insertOperation(oper);

    std::vector<Tensor::SPtr> grads = oper->getOutputs();
    return {{inputs[0], grads[0]}, {inputs[1], grads[1]}};
}

}  // namespace layers

Tensor::SPtr matmul(Tensor::SPtr m1, Tensor::SPtr m2)
{
    if (m1->shape().size() != 2 ||
        m2->shape().size() != 2)
        throw errors::NotMatchingShapesError();
    else if (m1->shape()[1] != m2->shape()[0])
        throw errors::NotMatchingShapesError();

    Oper::SPtr oper = std::make_shared<layers::MatmulOper>(m1, m2);
    getDefaultGraph()->insertOperation(oper);
    return oper->getOutputs()[0];
}

}  // namespace core

ITensorSPtr matmul(ITensorSPtr m1, ITensorSPtr m2)
{
    core::Tensor::SPtr mat1 = std::static_pointer_cast<core::Tensor>(m1);
    core::Tensor::SPtr mat2 = std::static_pointer_cast<core::Tensor>(m2);
    return ITensorSPtr(core::matmul(mat1, mat2));
}

}  // namespace dll
