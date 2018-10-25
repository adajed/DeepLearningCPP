#include "gradientBuilder.h"
#include "addOper.h"
#include "constantOper.h"
#include "gradientOper.h"
#include "graph.h"
#include "mulOper.h"

namespace dll
{
namespace core
{
GradientBuilder::GradientBuilder(Tensor::SPtr tensor)
    : mTensor(tensor), mTensorGradients(), mOperGradients()
{
}

GradientBuilder::TensorMap GradientBuilder::createGradients()
{
    // create gradients for each weights
    // d(weights)/d(weights) = constant(1., weights.shape)
    mTensorGradients.clear();
    for (auto pair : getDefaultGraph()->getWeights())
    {
        Tensor::SPtr weights = std::static_pointer_cast<Tensor>(pair.second);
        TensorMap weightGrads({{weights, constant(1., weights->shape())}});
        mTensorGradients.insert({weights, weightGrads});
    }

    return createGradientsForTensor(mTensor);
}

GradientBuilder::TensorMap GradientBuilder::createGradientsForTensor(
    Tensor::SPtr tensor)
{
    if (mTensorGradients.count(tensor) > 0) return mTensorGradients[tensor];

    Oper::SPtr oper = tensor->getOper();
    if (!oper->hasGradient())
    {
        mTensorGradients[tensor] = {};
        return mTensorGradients[tensor];
    }

    std::vector<Tensor::SPtr> inputs = oper->getInputs();

    // calculate TensorMap for inputs to the oper
    std::map<Tensor::SPtr, TensorMap> inputGradients;
    for (Tensor::SPtr input : inputs)
        inputGradients[input] = createGradientsForTensor(input);

    // calculate gradients for oper
    std::shared_ptr<GradientOper> gOper =
        std::static_pointer_cast<GradientOper>(oper);
    if (mOperGradients.count(oper) == 0)
        mOperGradients[oper] = gOper->gradients();
    TensorMap operGradients = mOperGradients[oper][tensor];

    // combine oper gradients with input gradients
    TensorMap tensorGrads;
    for (std::pair<std::string, ITensorSPtr> pair :
         getDefaultGraph()->getWeights())
    {
        Tensor::SPtr weights = std::static_pointer_cast<Tensor>(pair.second);
        Tensor::SPtr sum{nullptr};

        for (Tensor::SPtr input : inputs)
        {
            if (inputGradients[input].count(weights) > 0)
            {
                if (sum)
                    sum = sum +
                          inputGradients[input][weights] * operGradients[input];
                else
                    sum = inputGradients[input][weights] * operGradients[input];
            }
        }

        // TODO: is sum is nullptr it should be set to zero
        if (sum) tensorGrads[weights] = sum;
    }

    mTensorGradients[tensor] = tensorGrads;
    return tensorGrads;
}

}  // namespace core

std::map<ITensorSPtr, ITensorSPtr> gradients(ITensorSPtr tensor)
{
    using namespace core;
    Tensor::SPtr t = std::static_pointer_cast<Tensor>(tensor);
    GradientBuilder builder(t);
    GradientBuilder::TensorMap grads = builder.createGradients();

    // cast all Tensor::SPtr to ITensorSPtr
    std::map<ITensorSPtr, ITensorSPtr> rGrads;
    for (auto pair : core::getDefaultGraph()->getWeights())
    {
        Tensor::SPtr weights = std::static_pointer_cast<Tensor>(pair.second);
        if (grads.count(weights) > 0)
            rGrads[ITensorSPtr(weights)] = ITensorSPtr(grads[weights]);
        else
            rGrads[ITensorSPtr(weights)] =
                dll::constant(0., weights->getShape());
    }
    return rGrads;
}

}  // namespace dll
