#include "batchNorm.h"

#include "abstractTensor.h"
#include "activation.h"
#include "elementwise.h"
#include "graph.h"
#include "graphdl_ops.h"
#include "reduce.h"

#include <cmath>

namespace graphdl
{
namespace core
{
namespace layers
{
namespace
{
#define EPS 10e-8

size_t getBatchSize(const Tensor::SPtr& tensor, int numAxes)
{
    TensorShape shape = tensor->getShape();
    size_t size = 1;
    for (int i = 0; i < numAxes; ++i) size *= shape[i];
    return size;
}

size_t getFeatureSize(const Tensor::SPtr& tensor, int numAxes)
{
    return tensor->getShape().getCount() / getBatchSize(tensor, numAxes);
}

}  // namespace

void runBatchNormHost(const float* x, const float* alpha, const float* beta,
                      float* y, float* mean, float* stddev, size_t size,
                      size_t batchSize)
{
    size_t stride = size / batchSize;

    for (int i = 0; i < stride; ++i)
    {
        float val = 0.;
        for (int j = 0; j < batchSize; ++j) val += x[j * stride + i];
        mean[i] = val / float(batchSize);
    }

    for (int i = 0; i < stride; ++i)
    {
        float val = 0.;
        for (int j = 0; j < batchSize; ++j)
        {
            float f = x[j * stride + i] - mean[i];
            val += f * f;
        }
        stddev[i] = val / float(batchSize);
    }

    for (int i = 0; i < stride; ++i)
    {
        for (int j = 0; j < batchSize; ++j)
        {
            float val = x[j * stride + i] - mean[i];
            val /= std::sqrt(stddev[i] + EPS);
            val *= alpha[i];
            val += beta[i];
            y[j * stride + i] = val;
        }
    }
}

void runBatchNormGradientHost(const float* x, const float* alpha,
                              const float* beta, const float* y,
                              const float* yGrad, const float* mean,
                              const float* stddev, float* xGrad,
                              float* alphaGrad, float* betaGrad, size_t size,
                              size_t batchSize)
{
    size_t stride = size / batchSize;

    // betaGrad
    for (int i = 0; i < stride; ++i)
    {
        float val = 0.;
        for (int j = i; j < size; j += stride) val += yGrad[j];
        betaGrad[i] = val;
    }

    // alphaGrad
    for (int i = 0; i < stride; ++i)
    {
        float val = 0.;
        for (int j = i; j < size; j += stride) val += yGrad[j] * y[j];
        alphaGrad[i] = (val - betaGrad[i] * beta[i]) / alpha[i];
    }

    // xGrad
    for (int i = 0; i < stride; ++i)
    {
        float val = 0.;
        for (int j = i; j < size; j += stride) val += yGrad[j] * x[j];
        val -= betaGrad[i] * mean[i];

        for (int j = i; j < size; j += stride)
        {
            xGrad[j] = yGrad[j] - betaGrad[i] / float(batchSize) -
                       0.5 * (x[j] - mean[i]) * val / (stddev[i] + EPS);
            xGrad[j] /= std::sqrt(stddev[i] + EPS);
            xGrad[j] *= alpha[i];
        }
    }
}

BatchNormLayer::BatchNormLayer(ID id, const Tensor::SPtr& tensor,
                               const Tensor::SPtr& alpha,
                               const Tensor::SPtr& beta, int numAxes)
    : DifferentiableLayer(
          id, {tensor, alpha, beta},
          {createTensor("", tensor->getShape(), tensor->getType())}),
      mNumAxes(numAxes),
      mMean(tensor->getType(), getFeatureSize(tensor, numAxes)),
      mStddev(tensor->getType(), getFeatureSize(tensor, numAxes))
{
}

Layer::TensorMap BatchNormLayer::gradients(Tensor::SPtr out,
                                           Tensor::SPtr outGrad)
{
    assert(mOutputs[0] == out);

    Tensor::SPtr tensor = getInputs()[0];
    Tensor::SPtr alpha = getInputs()[1];
    Tensor::SPtr beta = getInputs()[2];

    Layer::SPtr layer = createLayer<BatchNormGradientLayer>(
        tensor, alpha, beta, out, outGrad, mNumAxes, &mMean, &mStddev);

    return {{tensor, layer->getOutputs()[0]},
            {alpha, layer->getOutputs()[1]},
            {beta, layer->getOutputs()[2]}};
}

void BatchNormLayer::execute(const std::vector<float*>& inputs,
                             const std::vector<float*>& outputs,
                             const InputDict& /*inputDict*/)
{
    float* x = inputs[0];
    float* alpha = inputs[1];
    float* beta = inputs[2];
    float* mean = mMean.getValues();
    float* stddev = mStddev.getValues();
    float* y = outputs[0];

    size_t size = getInputs()[0]->getShape().getCount();
    size_t batchSize = getBatchSize(getInputs()[0], mNumAxes);

    if (getInputs()[0]->getType() == MemoryType::kHOST_MEMORY)
        runBatchNormHost(x, alpha, beta, y, mean, stddev, size, batchSize);
#ifdef CUDA_AVAILABLE
    else
        cuda::runBatchNormDevice(x, alpha, beta, y, mean, stddev, size,
                                 batchSize);
#endif
}

void BatchNormLayer::initialize()
{
    mMean.allocate();
    mStddev.allocate();
}

BatchNormLayer::~BatchNormLayer()
{
    mMean.free();
    mStddev.free();
}

BatchNormGradientLayer::BatchNormGradientLayer(
    ID id, const Tensor::SPtr& tensor, const Tensor::SPtr& alpha,
    const Tensor::SPtr& beta, const Tensor::SPtr& out,
    const Tensor::SPtr& outGrad, int numAxes, Memory<float>* mean,
    Memory<float>* stddev)
    : Layer(id, {tensor, alpha, beta, out, outGrad},
            {createTensor("", tensor->getShape(), tensor->getType()),
             createTensor("", alpha->getShape(), alpha->getType()),
             createTensor("", beta->getShape(), beta->getType())}),
      mNumAxes(numAxes),
      mMean(mean),
      mStddev(stddev)
{
}

void BatchNormGradientLayer::execute(const std::vector<float*>& inputs,
                                     const std::vector<float*>& outputs,
                                     const InputDict& /*inputDict*/)
{
    float* x = inputs[0];
    float* alpha = inputs[1];
    float* beta = inputs[2];
    float* y = inputs[3];
    float* yGrad = inputs[4];
    float* xGrad = outputs[0];
    float* alphaGrad = outputs[1];
    float* betaGrad = outputs[2];
    float* mean = mMean->getValues();
    float* stddev = mStddev->getValues();

    size_t size = getInputs()[0]->getShape().getCount();
    size_t batchSize = getBatchSize(getInputs()[0], mNumAxes);

    if (getInputs()[0]->getType() == MemoryType::kHOST_MEMORY)
        runBatchNormGradientHost(x, alpha, beta, y, yGrad, mean, stddev, xGrad,
                                 alphaGrad, betaGrad, size, batchSize);
#ifdef CUDA_AVAILABLE
    else
        cuda::runBatchNormGradientDevice(x, alpha, beta, y, yGrad, mean, stddev,
                                         xGrad, alphaGrad, betaGrad, size,
                                         batchSize);
#endif
}

}  // namespace layers

Tensor::SPtr batchNorm(const Tensor::SPtr& tensor, const Tensor::SPtr& alpha,
                       const Tensor::SPtr& beta, int numAxes)
{
    if (numAxes <= 0) numAxes = tensor->getShape().size();

    Layer::SPtr layer =
        createLayer<layers::BatchNormLayer>(tensor, alpha, beta, numAxes);

    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr batchNorm(const ITensorPtr& tensor, const ITensorPtr& alpha,
                     const ITensorPtr& beta, int numAxes)
{
    core::Tensor::SPtr t = core::castITensorPtr(tensor)->get();
    core::Tensor::SPtr a = core::castITensorPtr(alpha)->get();
    core::Tensor::SPtr b = core::castITensorPtr(beta)->get();
    return core::makeAbstractTensor(core::batchNorm(t, a, b, numAxes));
}

}  // namespace graphdl
