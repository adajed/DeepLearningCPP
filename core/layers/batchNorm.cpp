#include "graphdl_ops.h"

#include  "abstractTensor.h"
#include "graph.h"

#include "activation.h"
#include "elementwise.h"
#include "reduceSum.h"

namespace graphdl
{
namespace core
{

Tensor::SPtr batchNorm(const Tensor::SPtr& tensor, const Tensor::SPtr& alpha,
                       const Tensor::SPtr& beta, int numAxes)
{
    if (numAxes <= 0) numAxes = tensor->getShape().size();

    Tensor::SPtr mean = reduceMean(tensor, numAxes);
    Tensor::SPtr t1 = elementwiseFront(tensor, mean, layers::Elementwise::kSUB);
    Tensor::SPtr stddev = reduceMean(square(t1), numAxes);
    Tensor::SPtr t2 = elementwiseFront(t1, sqrt(stddev) + 10e-6, layers::Elementwise::kDIV);
    Tensor::SPtr t3 = elementwiseFront(alpha, t2, layers::Elementwise::kMUL);
    return elementwiseFront(t3, beta, layers::Elementwise::kADD);
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
