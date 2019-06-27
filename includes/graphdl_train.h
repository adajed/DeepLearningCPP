//! \file graphdl_train.h
//! \brief Header file with available optimalization algorithms.
//!
//! \author Adam Jędrych adam.jedrych25@gmail.com
//!
#ifndef GRAPHDL_TRAIN_H_
#define GRAPHDL_TRAIN_H_

#include "graphdl.h"

//! \namespace grapdl
namespace graphdl
{
//! \namespace train
namespace train
{
//! \class ITrainer
//! \brief Interface for optimalization algorithm.
//!
class ITrainer
{
  public:
    //! \fn optimize
    //! \brief Creates operation that optimizes given value.
    //! \details Input tensor must be a scalar.
    //! \param tensor Value to optimize.
    //! \return Meta-operation that calculates gradients with respect
    //!     to the weights of the graph and modifies weights.
    //!
    virtual ITensorPtr optimize(const ITensorPtr& tensor) const = 0;

    virtual ~ITrainer() {}
};

//! \typedef ITrainerPtr
//! \brief Shared pointer for ITrainer.
//!
using ITrainerPtr = std::unique_ptr<ITrainer>;

//! \fn ITrainerPtr gradientDescent(float lr)
//! \brief Returns ITrainer representing vanilla gradient descent algorithm.
//! \param lr Learning rate for the algorithm.
//!
ITrainerPtr gradientDescent(float lr);

//! \fn ITrainerPtr momentum(float lr, float m)
//! \brief Returns ITrainer representing gradient descent algorithm with
//!                momentum.
//! \param lr Learning rate for the algorithm.
//! \param m Momentum value.
//!
ITrainerPtr momentum(float lr, float m);

//! \fn ITrainerPtr adam(float lr, float beta1, float beta2, float eps)
//! \brief Returns ITrainer representing ADAM algorithm.
//! \param lr Learning rate.
//! \param beta1 Beta1.
//! \param beta2 Beta2.
//! \param eps Epsilon.
//!
ITrainerPtr adam(float lr, float beta1, float beta2, float eps);

//! \fn ITrainerPtr adagrad(float lr, float eps)
//! \brief Returns trainer representing adagrad algorithm.
//! \param lr Learning rate.
//! \param eps Epsilon.
//!
ITrainerPtr adagrad(float lr, float eps);

}  // namespace train
}  // namespace graphdl

#endif  // GRAPHDL_TRAIN_H_
