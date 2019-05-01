#include "abstractTensor.h"
#include "activation.h"
#include "elementwise.h"
#include "graph.h"
#include "graphdl_ops.h"
#include "reduce.h"

namespace graphdl
{
namespace core
{
Tensor::SPtr batchNorm(const Tensor::SPtr& tensor, const Tensor::SPtr& alpha,
                       const Tensor::SPtr& beta, int numAxes)
{
    if (numAxes <= 0) numAxes = tensor->getShape().size();

    int reduceSize = tensor->getShape().subshape(0, numAxes).getCount();
    Tensor::SPtr mean = reduceFront(tensor, numAxes, layers::ReduceType::kSUM);
    mean = mean / float(reduceSize);
    Tensor::SPtr t1 = elementwiseBack(tensor, mean, layers::Elementwise::kSUB);
    Tensor::SPtr stddev =
        reduceFront(square(t1), numAxes, layers::ReduceType::kSUM);
    stddev = stddev / float(reduceSize);
    Tensor::SPtr t2 =
        elementwiseBack(t1, sqrt(stddev) + 10e-6, layers::Elementwise::kDIV);
    Tensor::SPtr t3 = elementwiseBack(alpha, t2, layers::Elementwise::kMUL);
    return elementwiseBack(t3, beta, layers::Elementwise::kADD);
}

}  // namespace core

ITensorPtr batchNorm(const ITensorPtr& tensor, const ITensorPtr& alpha,
                     const ITensorPtr& beta, int numAxes)
{
    core::Tensor::SPtr t = core::castITensorPtr(tensor)->get();
    core::Tensor::SPtr a = core::castITensorPtr(alpha)->get();
    core::Tensor::SPtr b = core::castITensorPtr(beta)->get();
    return core::makeAbstractTensor(core::batchNorm(t, a, b, numAxes));
}

}  // namespace graphdl
