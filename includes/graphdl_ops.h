#ifndef GRAPHDL_OPS_H_
#define GRAPHDL_OPS_H_

#include "graphdl.h"

namespace graphdl
{
ITensorPtr add(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr operator+(const ITensorPtr& t1, const ITensorPtr& t2);

ITensorPtr sub(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr operator-(const ITensorPtr& t1, const ITensorPtr& t2);

ITensorPtr mul(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr operator*(const ITensorPtr& t1, const ITensorPtr& t2);

ITensorPtr div(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr operator/(const ITensorPtr& t1, const ITensorPtr& t2);

ITensorPtr constant(float value, const Shape& shape, MemoryLocation location);

ITensorPtr matmul(const ITensorPtr& m1, const ITensorPtr& m2);

ITensorPtr relu(const ITensorPtr& t);
ITensorPtr sigmoid(const ITensorPtr& t);
ITensorPtr tanh(const ITensorPtr& t);
ITensorPtr square(const ITensorPtr& t);
ITensorPtr abs(const ITensorPtr& t);
ITensorPtr neg(const ITensorPtr& t);
ITensorPtr reciprocal(const ITensorPtr& t);
ITensorPtr log(const ITensorPtr& t);

ITensorPtr assign(const ITensorPtr& dest, const ITensorPtr& src);

ITensorPtr reduceSum(const ITensorPtr& t);

ITensorPtr addN(std::vector<ITensorPtr> tensors);

ITensorPtr group(const std::vector<ITensorPtr>& tensors);

}  // namespace graphdl

#endif  // GRAPHDL_OPS_H_
