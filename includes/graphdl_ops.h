#ifndef GRAPHDL_OPS_H_
#define GRAPHDL_OPS_H_

#include "graphdl.h"

namespace graphdl
{
ITensorPtr add(ITensorPtr t1, ITensorPtr t2);
ITensorPtr operator+(ITensorPtr t1, ITensorPtr t2);

ITensorPtr sub(ITensorPtr t1, ITensorPtr t2);
ITensorPtr operator-(ITensorPtr t1, ITensorPtr t2);

ITensorPtr mul(ITensorPtr t1, ITensorPtr t2);
ITensorPtr operator*(ITensorPtr t1, ITensorPtr t2);

ITensorPtr div(ITensorPtr t1, ITensorPtr t2);
ITensorPtr operator/(ITensorPtr t1, ITensorPtr t2);

ITensorPtr constant(float value, const Shape& shape);

ITensorPtr matmul(ITensorPtr m1, ITensorPtr m2);

ITensorPtr relu(ITensorPtr t);
ITensorPtr sigmoid(ITensorPtr t);
ITensorPtr tanh(ITensorPtr t);
ITensorPtr square(ITensorPtr t);
ITensorPtr abs(ITensorPtr t);
ITensorPtr neg(ITensorPtr t);
ITensorPtr reciprocal(ITensorPtr t);
ITensorPtr log(ITensorPtr t);

ITensorPtr assign(ITensorPtr dest, ITensorPtr src);

ITensorPtr reduceSum(ITensorPtr tensor);

ITensorPtr addN(std::vector<ITensorPtr> tensors);

}  // namespace graphdl

#endif  // GRAPHDL_OPS_H_
