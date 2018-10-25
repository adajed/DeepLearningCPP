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

}  // namespace dll

#endif  // DLL_OPS_H_
