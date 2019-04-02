#ifndef GRAPHDL_OPS_H_
#define GRAPHDL_OPS_H_

#include "graphdl.h"

namespace graphdl
{
ITensorPtr add(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr add(float val, const ITensorPtr& t2);
ITensorPtr add(const ITensorPtr& t1, float val);
ITensorPtr operator+(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr operator+(float val, const ITensorPtr& t2);
ITensorPtr operator+(const ITensorPtr& t1, float val);

ITensorPtr sub(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr sub(float val, const ITensorPtr& t2);
ITensorPtr sub(const ITensorPtr& t1, float val);
ITensorPtr operator-(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr operator-(float val, const ITensorPtr& t2);
ITensorPtr operator-(const ITensorPtr& t1, float val);

ITensorPtr mul(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr mul(float val, const ITensorPtr& t2);
ITensorPtr mul(const ITensorPtr& t1, float val);
ITensorPtr operator*(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr operator*(float val, const ITensorPtr& t2);
ITensorPtr operator*(const ITensorPtr& t1, float val);

ITensorPtr div(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr div(float val, const ITensorPtr& t2);
ITensorPtr div(const ITensorPtr& t1, float val);
ITensorPtr operator/(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr operator/(float val, const ITensorPtr& t2);
ITensorPtr operator/(const ITensorPtr& t1, float val);

ITensorPtr constant(float value, const Shape& shape, MemoryLocation location);
ITensorPtr scalar(float value, MemoryLocation location);

ITensorPtr matmul(const ITensorPtr& m1, const ITensorPtr& m2);

ITensorPtr relu(const ITensorPtr& t);
ITensorPtr sigmoid(const ITensorPtr& t);
ITensorPtr tanh(const ITensorPtr& t);
ITensorPtr square(const ITensorPtr& t);
ITensorPtr abs(const ITensorPtr& t);
ITensorPtr neg(const ITensorPtr& t);
ITensorPtr reciprocal(const ITensorPtr& t);
ITensorPtr log(const ITensorPtr& t);
ITensorPtr sqrt(const ITensorPtr& t);
ITensorPtr exp(const ITensorPtr& t);

ITensorPtr assign(const ITensorPtr& dest, const ITensorPtr& src);

ITensorPtr reduceSum(const ITensorPtr& t, int numAxes = -1);

ITensorPtr addN(std::vector<ITensorPtr> tensors);

ITensorPtr group(const std::vector<ITensorPtr>& tensors);

ITensorPtr queue(const std::vector<ITensorPtr>& tensors);

ITensorPtr reshape(const ITensorPtr& t, const Shape& shape);

ITensorPtr maxPool2D(const ITensorPtr& tensor,
                     const std::vector<int>& kernel = {2, 2},
                     const std::vector<int>& strides = {2, 2},
                     const std::string& padding = "VALID",
                     const std::string& format = "NHWC");

ITensorPtr avgPool2D(const ITensorPtr& tensor,
                     const std::vector<int>& kernel = {2, 2},
                     const std::vector<int>& strides = {2, 2},
                     const std::string& padding = "VALID",
                     const std::string& format = "NHWC");

ITensorPtr conv2D(const ITensorPtr& tensor, const ITensorPtr& kernel,
                  const std::vector<int>& strides, const std::string& padding);

ITensorPtr softmax(const ITensorPtr& tensor, int numAxes = -1);

}  // namespace graphdl

#endif  // GRAPHDL_OPS_H_
