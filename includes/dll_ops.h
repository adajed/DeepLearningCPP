#ifndef DLL_OPS_H_
#define DLL_OPS_H_

#include "dll.h"

namespace dll
{
ITensorSPtr add(ITensorSPtr t1, ITensorSPtr t2);
ITensorSPtr operator+(ITensorSPtr t1, ITensorSPtr t2);

ITensorSPtr sub(ITensorSPtr t1, ITensorSPtr t2);
ITensorSPtr operator-(ITensorSPtr t1, ITensorSPtr t2);

ITensorSPtr mul(ITensorSPtr t1, ITensorSPtr t2);
ITensorSPtr operator*(ITensorSPtr t1, ITensorSPtr t2);

ITensorSPtr div(ITensorSPtr t1, ITensorSPtr t2);
ITensorSPtr operator/(ITensorSPtr t1, ITensorSPtr t2);

ITensorSPtr constant(float value, const Shape& shape);

ITensorSPtr matmul(ITensorSPtr m1, ITensorSPtr m2);

ITensorSPtr relu(ITensorSPtr t);
ITensorSPtr sigmoid(ITensorSPtr t);
ITensorSPtr tanh(ITensorSPtr t);
ITensorSPtr square(ITensorSPtr t);
ITensorSPtr abs(ITensorSPtr t);
ITensorSPtr neg(ITensorSPtr t);
ITensorSPtr reciprocal(ITensorSPtr t);

}  // namespace dll

#endif  // DLL_OPS_H_
