//! \file graphdl_ops.h
//! \brief Header file with available operations.
//!
//! \author Adam Jędrych adam.jedrych25@gmail.com
//!
#ifndef GRAPHDL_OPS_H_
#define GRAPHDL_OPS_H_

#include "graphdl.h"

//! \namespace graphdl
namespace graphdl
{
//! \name Pointwise addition
//! \details If shapes of tensors don't match it tries to broadcast
//!     one to another. If one of the inputs is float then it
//!     considered as constant scalar.
///@{
//!
ITensorPtr add(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr add(float val, const ITensorPtr& t2);
ITensorPtr add(const ITensorPtr& t1, float val);
ITensorPtr operator+(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr operator+(float val, const ITensorPtr& t2);
ITensorPtr operator+(const ITensorPtr& t1, float val);
///@}

//! \name Pointwise substraction.
//! \details If shapes of tensors don't match it tries to broadcast
//!     one to another. If one of the inputs is float then it
//!     considered as constant scalar.
///@{
//!
ITensorPtr sub(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr sub(float val, const ITensorPtr& t2);
ITensorPtr sub(const ITensorPtr& t1, float val);
ITensorPtr operator-(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr operator-(float val, const ITensorPtr& t2);
ITensorPtr operator-(const ITensorPtr& t1, float val);
///@}

//! \name Pointwise multiplication.
//! \details If shapes of tensors don't match it tries to broadcast
//!     one to another. If one of the inputs is float then it
//!     considered as constant scalar.
///@{
//!
ITensorPtr mul(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr mul(float val, const ITensorPtr& t2);
ITensorPtr mul(const ITensorPtr& t1, float val);
ITensorPtr operator*(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr operator*(float val, const ITensorPtr& t2);
ITensorPtr operator*(const ITensorPtr& t1, float val);
///@}

//! \name Pointwise division.
//! \details If shapes of tensors don't match it tries to broadcast
//!     one to another. If one of the inputs is float then it
//!     considered as constant scalar.
///@{
//!
ITensorPtr div(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr div(float val, const ITensorPtr& t2);
ITensorPtr div(const ITensorPtr& t1, float val);
ITensorPtr operator/(const ITensorPtr& t1, const ITensorPtr& t2);
ITensorPtr operator/(float val, const ITensorPtr& t2);
ITensorPtr operator/(const ITensorPtr& t1, float val);
///@}

//! \fn ITensorPtr constant(float value, const Shape& shape, MemoryLocation
//! location) \brief Creates constant tensor.
//!
ITensorPtr constant(float value, const Shape& shape, MemoryLocation location);

//! \fn ITensorPtr scalar(float value, MemoryLocation location)
//! \brief Creates constant scalar (tensor of dimension 0).
//!
ITensorPtr scalar(float value, MemoryLocation location);

//! \fn ITensorPtr matmul(const ITensorPtr& m1, const ITensorPtr& m2)
//! \brief Matrix multiplication.
//! Input tensors must be 2-dimensional.
//! Output tensor is also 2-dimensional.
//!
ITensorPtr matmul(const ITensorPtr& m1, const ITensorPtr& m2);

//! \name Pointwise operations
//!
///@{
//!

//! \fn ITensorPtr relu(const ITensorPtr& t)
//! \brief Applies pointwise relu function.
//!
ITensorPtr relu(const ITensorPtr& t);

//! \fn ITensorPtr sigmoid(const ITensorPtr& t)
//! \brief Applies pointwise sigmoid function.
//!
ITensorPtr sigmoid(const ITensorPtr& t);

//! \fn ITensorPtr tanh(const ITensorPtr& t)
//! \brief Applies pointwise tanh function.
//!
ITensorPtr tanh(const ITensorPtr& t);

//! \fn ITensorPtr square(const ITensorPtr& t)
//! \brief Squares each value in tensor.
//!
ITensorPtr square(const ITensorPtr& t);

//! \fn ITensorPtr abs(const ITensorPtr& t)
//! \brief Applies pointwise abs function.
//!
ITensorPtr abs(const ITensorPtr& t);

//! \fn ITensorPtr neg(const ITensorPtr& t)
//! \brief Negates each value in tensor.
//!
ITensorPtr neg(const ITensorPtr& t);

//! \fn ITensorPtr reciprocal(const ITensorPtr& t)
//! \brief Reverses each number in tensor.
//!
ITensorPtr reciprocal(const ITensorPtr& t);

//! \fn ITensorPtr log(const ITensorPtr& t)
//! \brief Calculates pointwise natural logarithm.
//!
ITensorPtr log(const ITensorPtr& t);

//! \fn ITensorPtr sqrt(const ITensorPtr& t)
//! \brief Calculates pointwise square root.
//!
ITensorPtr sqrt(const ITensorPtr& t);

//! \fn ITensorPtr exp(const ITensorPtr& t)
//! \brief Calculates pointwise exponent.
//!
ITensorPtr exp(const ITensorPtr& t);

///@}

//! \fn ITensorPtr assign(const ITensorPtr& dest, const ITensorPtr& src)
//! \brief Sets dest to value in src, returns empty tensor.
//! Note that if dest tensor is not weights then its content
//!     will be reseted at the next computation.
//!
ITensorPtr assign(const ITensorPtr& dest, const ITensorPtr& src);

//! \fn ITensorPtr reduceSum(const ITensorPtr& t, int numAxes = -1)
//! \brief Calculate sum of values along last numAxes axes.
//! If numAxes is non-positive calculates sum of all values in tensor.
//!
ITensorPtr reduceSum(const ITensorPtr& t, int numAxes = -1);

ITensorPtr reduceMax(const ITensorPtr& t, int numAxes = -1);

ITensorPtr reduceMin(const ITensorPtr& t, int numAxes = -1);

//! \fn ITensorPtr addN(std::vector<ITensorPtr> tensors)
//! \brief Calculates pointwise addtion of all tensors in vector.
//! All tensors have to be the same size.
//!
ITensorPtr addN(std::vector<ITensorPtr> tensors);

//! \fn ITensorPtr group(const std::vector<ITensorPtr>& tensors)
//! \brief Groups operations in single meta-operation.
//! Output of this operation is empty tensor.
//! It was created for grouping mutliple assing operations.
//! Doesn't ensure anything about order of execution,
//!     for this see queue.
//!
//! \see queue
//!
ITensorPtr group(const std::vector<ITensorPtr>& tensors);

//! \fn ITensorPtr queue(const std::vector<ITensorPtr>& tensors)
//! \brief Groups operations in single meta-operation.
//! Same as group, but ensures that operations will be executed in order.
//!
//! \see group
//!
ITensorPtr queue(const std::vector<ITensorPtr>& tensors);

//! \fn ITensorPtr reshape(const ITensorPtr& t, const Shape& shape)
//! \brief Creates new tensor with different shape.
//! Shape must contain the same number of element as tensor.
//! Element are moved in row-major order.
//!
ITensorPtr reshape(const ITensorPtr& t, const Shape& shape);

//! \fn ITensorPtr maxPool2D(const ITensorPtr& tensor,
//!                          const std::vector<int>& kernel = {2, 2},
//!                          const std::vector<int>& strides = {2, 2},
//!                          const std::string& padding = "VALID",
//!                          const std::string& format = "NHWC")
//! \brief Applies max pooling operation.
//! \param tensor Tensor on which pooling will be performed, must be
//! 4-dimensional. \param kernel Kernel shape, should be of length 1 or 2.
//! \param strides Strides, should be of length 1 or 2.
//! \param padding Type of padding, should be one of: "SAME", "VALID".
//! \param format Type of data format, should be one of : "NHWC", "NCHW".
//!
ITensorPtr maxPool2D(const ITensorPtr& tensor,
                     const std::vector<int>& kernel = {2, 2},
                     const std::vector<int>& strides = {2, 2},
                     const std::string& padding = "VALID",
                     const std::string& format = "NHWC");

//! \fn ITensorPtr avgPool2D(const ITensorPtr& tensor,
//!                          const std::vector<int>& kernel = {2, 2},
//!                          const std::vector<int>& strides = {2, 2},
//!                          const std::string& padding = "VALID",
//!                          const std::string& format = "NHWC")
//! \brief Applies average pooling operation.
//! \param tensor Tensor on which pooling will be performed, must be
//! 4-dimensional. \param kernel Kernel shape, should be of length 1 or 2.
//! \param strides Strides, should be of length 1 or 2.
//! \param padding Type of padding, should be one of: "SAME", "VALID".
//! \param format Type of data format, should be one of : "NHWC", "NCHW".
//!
ITensorPtr avgPool2D(const ITensorPtr& tensor,
                     const std::vector<int>& kernel = {2, 2},
                     const std::vector<int>& strides = {2, 2},
                     const std::string& padding = "VALID",
                     const std::string& format = "NHWC");

//! \fn ITensorPtr conv2D(const ITensorPtr& tensor, const ITensorPtr& kernel,
//!                       const std::vector<int>& strides = {1, 1},
//!                       const std::string& padding = "SAME",
//!                       const std::string& format = "NHWC")
//! \brief Applies 2D convolution operation.
//! \param tensor Tensor on which convolution will be performed, must be
//! 4-dimensional. \param kernel Tensor with convolution kernel, must be
//! 4-dimensional. \param strides Strides, should be of length 1 or 2. \param
//! padding Type of padding, should be one of: "SAME", "VALID". \param format
//! Type of data format, should be one of : "NHWC", "NCHW".
//!
ITensorPtr conv2D(const ITensorPtr& tensor, const ITensorPtr& kernel,
                  const std::vector<int>& strides = {1, 1},
                  const std::string& padding = "SAME",
                  const std::string& format = "NHWC");

//! \fn ITensorPtr nhwc2nchw(const ITensorPtr& tensor)
//! \brief Reformats tensor from NHWC to NCHW format.
//!
ITensorPtr nhwc2nchw(const ITensorPtr& tensor);

//! \fn ITensorPtr nchw2nhwc(const ITensorPtr& tensor)
//! \brief Reformats tensor from NCHW to NHWC format.
//!
ITensorPtr nchw2nhwc(const ITensorPtr& tensor);

//! \fn ITensorPtr softmax(const ITensorPtr& tensor, int numAxes = -1)
//! \brief Applies softmax operation on last numAxes axes.
//! \param tensor Tensor on which softmax will be performed.
//! \param numAxes Number of axes, if non-positive then it is set
//!                to number of dimenstions of tensor.
//!
ITensorPtr softmax(const ITensorPtr& tensor, int numAxes = -1);

//! \fn ITensorPtr batchNorm(const ITensorPtr& tensor, const ITensorPtr& alpha,
//!                          const ITensorPtr& beta, int numAxes = -1)
//! \brief Applies batch normalization.
//! \param tensor Tensor on which normalization will be performed.
//! \param alpha Tensor with alpha weights for normalization.
//! \param beta Tensor with beta weights for normalization.
//! \param numAxes Number of axes (counting from begging) normalization
//!                will be applied.
//! \details If tensor has N dimensions, then alpha and beta should have the
//! shape
//!          as last <N - numAxes> of axes of tensor.
//!          If numAxes is non-positive then normalization if done over all
//!          axes.
//!
ITensorPtr batchNorm(const ITensorPtr& tensor, const ITensorPtr& alpha,
                     const ITensorPtr& beta, int numAxes = -1);

}  // namespace graphdl

#endif  // GRAPHDL_OPS_H_
