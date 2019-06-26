#ifndef SAMPLES_UTILS_H_
#define SAMPLES_UTILS_H_

#include "graphdl.h"
#include "graphdl_ops.h"

#include <iostream>
#include <numeric>

using namespace graphdl;

struct ComputationalGraph
{
    std::map<std::string, ITensorPtr> inputs;
    std::vector<ITensorPtr> weights;
    ITensorPtr output;
    ITensorPtr loss;
    ITensorPtr optimize;
};

int calcNumCorrect(const HostTensor& y, const HostTensor& pred, int n)
{
    int count = 0;
    for (int b = 0; b < n; ++b)
    {
        int bestY = 0, bestPred = 0;
        float maxY = y[10 * b], maxPred = pred[10 * b];
        for (int i = 1; i < 10; ++i)
        {
            if (y[10 * b + i] > maxY)
            {
                bestY = i;
                maxY = y[10 * b + i];
            }
            if (pred[10 * b + i] > maxPred)
            {
                bestPred = i;
                maxPred = pred[10 * b + i];
            }
        }

        if (bestY == bestPred) count++;
    }

    return count;
}

float mean(const std::vector<float>& v) { float m = std::accumulate(v.begin(), v.end(), 0.); return m / float(v.size()); }

float accuracy(const std::vector<int>& v, int n)
{
    int sum = std::accumulate(v.begin(), v.end(), 0);
    return float(sum) / float(v.size() * n);
}

ITensorPtr create_matmulAndAddBias(ITensorPtr tensor,
                                   int num_features,
                                   const std::string& name)
{
#ifdef CUDA_AVAILABLE
    MemoryLocation loc = MemoryLocation::kDEVICE;
#else
    MemoryLocation loc = MemoryLocation::kHOST;
#endif

    int in_features = tensor->getShape()[1];
    Shape weights_shape(2);
    weights_shape[0] = in_features;
    weights_shape[1] = num_features;

    Shape bias_shape(1);
    bias_shape[0] = num_features;

    IInitializerPtr init = uniformInitializer(-1., 1., 0);

    ITensorPtr weights = createWeights(name + "_weights", weights_shape, init, loc);
    ITensorPtr bias = createWeights(name + "_bias", bias_shape, init, loc);

    return matmul(tensor, weights) + bias;
}

ITensorPtr create_conv2D(ITensorPtr tensor,
                         int num_filters,
                         const std::vector<int>& kernel,
                         const std::vector<int>& strides,
                         const std::string& padding,
                         const std::string& format,
                         const std::string& name)
{
#ifdef CUDA_AVAILABLE
    MemoryLocation loc = MemoryLocation::kDEVICE;
#else
    MemoryLocation loc = MemoryLocation::kHOST;
#endif

    Shape tensor_shape = tensor->getShape();
    Shape weights_shape(4);

    weights_shape[0] = kernel[0];
    weights_shape[1] = kernel[1];
    if (format == "nhwc" || format == "NHWC")
        weights_shape[2] = tensor_shape[3];
    else if (format == "nchw" || format == "NCHW")
        weights_shape[2] = tensor_shape[1];
    else
        throw std::runtime_error("Wrong format string");
    weights_shape[3] = num_filters;

    int sum_channels = weights_shape[2] + weights_shape[3];
    IInitializerPtr init = normalInitializer(0., 1. / float(sum_channels), 0);
    ITensorPtr weights = createWeights(name + "_weights", weights_shape, init, loc);

    return conv2D(tensor, weights, strides, padding, format);
}

ITensorPtr create_batchNorm(ITensorPtr t, int numAxes, const std::string& name)
{
#ifdef CUDA_AVAILABLE
    MemoryLocation loc = MemoryLocation::kDEVICE;
#else
    MemoryLocation loc = MemoryLocation::kHOST;
#endif

    Shape shape = t->getShape();
    if (numAxes <= 0)
        numAxes = shape.size();

    Shape new_shape(shape.size() - numAxes);
    for (int i = numAxes; i < shape.size(); ++i)
        new_shape[i - numAxes] = shape[i];

    IInitializerPtr init_0 = constantInitializer(0.);
    IInitializerPtr init_1 = constantInitializer(1.);

    ITensorPtr alpha = createWeights(name + "_alpha", new_shape, init_1, loc);
    ITensorPtr beta = createWeights(name + "_beta", new_shape, init_0, loc);
    return batchNorm(t, alpha, beta, numAxes);
}

#endif  // SAMPLES_UTILS_H_
