#include "gradientBuilder.h"
#include "addN.h"
#include "constantOper.h"
#include "dll_errors.h"
#include "dll_ops.h"
#include "elementwiseOper.h"
#include "gradientOper.h"
#include "graph.h"

namespace dll
{
namespace core
{
GradientBuilder::GradientBuilder(Tensor::SPtr tensor)
    : mTensor(tensor),
      mTensorGradients(),
      mGradientsToCalc(),
      mCalculatedTensors()
{
    if (tensor->shape().count() != 1)
        throw errors::NotScalarGradientsCalculation();
}

void GradientBuilder::findTensorOutputs(Tensor::SPtr tensor,
                                        std::set<Tensor::SPtr>& visited)
{
    if (visited.count(tensor) > 0) return;
    visited.insert(tensor);

    Oper::SPtr oper = tensor->getOper();
    if (oper->hasGradient())
    {
        std::vector<Tensor::SPtr> inputs = oper->getInputs();
        for (Tensor::SPtr in : inputs)
        {
            if (mGradientsToCalc.count(in) == 0)
                mGradientsToCalc.insert({in, {}});
            mGradientsToCalc[in].insert(tensor);
            findTensorOutputs(in, visited);
        }
    }
}

GradientBuilder::TensorMap GradientBuilder::createGradients()
{
    std::set<Tensor::SPtr> visited;
    findTensorOutputs(mTensor, visited);

    mTensorGradients.clear();
    mCalculatedTensors.clear();
    //! d(mTensor)/d(mTensor) = 1.
    mCalculatedTensors.insert({mTensor, constant(1., mTensor->shape())});

    calculateGradientsForTensor(mTensor);

    TensorMap gradients;
    for (auto pair : getDefaultGraph()->getWeights())
    {
        Tensor::SPtr weights = std::static_pointer_cast<Tensor>(pair.second);
        if (mCalculatedTensors.count(weights) == 0)
            mCalculatedTensors.insert(
                {weights, constant(0., weights->shape())});
        gradients.insert({weights, mCalculatedTensors[weights]});
    }
    return gradients;
}

void GradientBuilder::modifyTensorGradient(Tensor::SPtr tensor,
                                           Tensor::SPtr tensorGrad)
{
    if (mTensorGradients.count(tensor) == 0)
        mTensorGradients.insert({tensor, {}});
    mTensorGradients[tensor].push_back(tensorGrad);

    if (mGradientsToCalc[tensor].empty())
        mCalculatedTensors[tensor] = core::addN(mTensorGradients[tensor]);
}

void GradientBuilder::calculateGradientsForTensor(Tensor::SPtr tensor)
{
    if (!mGradientsToCalc[tensor].empty()) return;
    Tensor::SPtr tensorGrad = mCalculatedTensors[tensor];

    Oper::SPtr oper = tensor->getOper();
    if (oper->hasGradient())
    {
        std::shared_ptr<GradientOper> g_oper =
            std::static_pointer_cast<GradientOper>(oper);

        std::vector<Tensor::SPtr> inputs = g_oper->getInputs();
        std::map<Tensor::SPtr, Tensor::SPtr> inputGrads =
            g_oper->gradients(tensor, tensorGrad);

        for (Tensor::SPtr in : inputs)
        {
            mGradientsToCalc[in].erase(tensor);
            modifyTensorGradient(in, inputGrads[in]);
            calculateGradientsForTensor(in);
        }
    }
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
