#include "convolution.h"

#include "abstractTensor.h"
#include "graph.h"

#include <cassert>
#include <utility>

namespace graphdl
{
namespace core
{
namespace layers
{
namespace
{
int ceil(int x, int y)
{
    return x / y + int(x % y > 0);
}

Tensor::SPtr createOutput(const Tensor::SPtr& t, const Tensor::SPtr& k,
                          const std::vector<int>& s, PaddingType padding)
{
    TensorShape shape = t->getShape();
    TensorShape kernel = k->getShape();

    shape[1] = kernel[0];
    if (padding == PaddingType::kVALID)
    {
        shape[2] = ceil(shape[2] - kernel[2] + 1, s[0]);
        shape[3] = ceil(shape[3] - kernel[3] + 1, s[1]);
    }
    else  // padding == PaddingType::kSAME
    {
        shape[2] = ceil(shape[2], s[0]);
        shape[3] = ceil(shape[3], s[1]);
    }

    return createTensor("", shape, t->getType());
}

float convReduce(const float* in, const float* k, const std::vector<int>& shape,
                 const std::vector<int>& kernel, int x, int y)
{
    float val = 0.;

    for (int iC = 0; iC < kernel[1]; ++iC)
    {
        for (int iX = x < 0 ? -x : 0; iX < kernel[2]; ++iX)
        {
            if (x + iX >= shape[2]) break;
            for (int iY = y < 0 ? -y : 0; iY < kernel[3]; ++iY)
            {
                if (y + iY >= shape[3]) break;
                val +=
                    in[(x + iX) * shape[3] + y + iY] * k[iX * kernel[3] + iY];
            }
        }
        in += shape[2] * shape[3];
        k += kernel[2] * kernel[3];
    }

    return val;
}

template <PaddingType padding>
void conv2D(const float* xArr, const float* kArr, float* yArr,
            const std::vector<int>& shape, const std::vector<int>& kernel,
            const std::vector<int>& strides)
{
    int outShape[] = {shape[0], kernel[0], 0, 0};
    if (padding == PaddingType::kVALID)
    {
        outShape[2] = ceil(shape[2] - kernel[2] + 1, strides[0]);
        outShape[3] = ceil(shape[3] - kernel[3] + 1, strides[1]);
    }
    else if (padding == PaddingType::kSAME)
    {
        outShape[2] = ceil(shape[2], strides[0]);
        outShape[3] = ceil(shape[3], strides[1]);
    }

    for (int n = 0; n < outShape[0]; ++n)
        for (int c = 0; c < outShape[1]; ++c)
            for (int x = 0; x < outShape[2]; ++x)
                for (int y = 0; y < outShape[3]; ++y)
                {
                    int x2 = x * strides[0], y2 = y * strides[1];
                    if (padding == PaddingType::kSAME)
                    {
                        x2 -= (kernel[2] - 1) / 2;
                        y2 -= (kernel[3] - 1) / 2;
                    }

                    size_t inPos = n * shape[1] * shape[2] * shape[3];
                    size_t kerPos = c * kernel[1] * kernel[2] * kernel[3];
                    size_t outPos = n * outShape[1] + c;
                    outPos = outPos * outShape[2] + x;
                    outPos = outPos * outShape[3] + y;
                    yArr[outPos] = convReduce(xArr + inPos, kArr + kerPos,
                                              shape, kernel, x2, y2);
                }
}

void convGradientReduce(const float* xArr, const float* kArr, float yG,
                        float* xGArr, float* kGArr,
                        const std::vector<int>& shape,
                        const std::vector<int>& kernel, int x, int y)
{
    for (int iC = 0; iC < kernel[1]; ++iC)
    {
        for (int iX = x < 0 ? -x : 0; iX < kernel[2]; ++iX)
        {
            if (x + iX >= shape[2]) break;
            for (int iY = y < 0 ? -y : 0; iY < kernel[3]; ++iY)
            {
                if (y + iY >= shape[3]) break;
                xGArr[(x + iX) * shape[3] + y + iY] +=
                    yG * kArr[iX * kernel[3] + iY];
                kGArr[iX * kernel[3] + iY] +=
                    yG * xArr[(x + iX) * shape[3] + y + iY];
            }
        }
        xArr += shape[2] * shape[3];
        xGArr += shape[2] * shape[3];
        kArr += kernel[2] * kernel[3];
        kGArr += kernel[2] * kernel[3];
    }
}

template <PaddingType padding>
void conv2DGradient(const float* xArr, const float* kArr, const float* yGArr,
                    float* xGArr, float* kGArr, const std::vector<int>& shape,
                    const std::vector<int>& kernel,
                    const std::vector<int>& strides)
{
    int outShape[] = {shape[0], kernel[0], 0, 0};
    if (padding == PaddingType::kVALID)
    {
        outShape[2] = ceil(shape[2] - kernel[2] + 1, strides[0]);
        outShape[3] = ceil(shape[3] - kernel[3] + 1, strides[1]);
    }
    else if (padding == PaddingType::kSAME)
    {
        outShape[2] = ceil(shape[2], strides[0]);
        outShape[3] = ceil(shape[3], strides[1]);
    }

    for (int pos = 0; pos < shape[0] * shape[1] * shape[2] * shape[3]; ++pos)
        xGArr[pos] = 0.;
    for (int pos = 0; pos < kernel[0] * kernel[1] * kernel[2] * kernel[3];
         ++pos)
        kGArr[pos] = 0.;

    for (int n = 0; n < outShape[0]; ++n)
        for (int c = 0; c < outShape[1]; ++c)
            for (int x = 0; x < outShape[2]; ++x)
                for (int y = 0; y < outShape[3]; ++y)
                {
                    int x2 = x * strides[0], y2 = y * strides[1];
                    if (padding == PaddingType::kSAME)
                    {
                        x2 -= (kernel[2] - 1) / 2;
                        y2 -= (kernel[3] - 1) / 2;
                    }

                    size_t inPos = n * shape[1] * shape[2] * shape[3];
                    size_t kerPos = c * kernel[1] * kernel[2] * kernel[3];
                    size_t outPos = n * outShape[1] + c;
                    outPos = outPos * outShape[2] + x;
                    outPos = outPos * outShape[3] + y;

                    convGradientReduce(xArr + inPos, kArr + kerPos,
                                       yGArr[outPos], xGArr + inPos,
                                       kGArr + kerPos, shape, kernel, x2, y2);
                }
}

void runConv2DHost(const float* x, const float* k, float* y,
                   const std::vector<int>& shape,
                   const std::vector<int>& kernel,
                   const std::vector<int>& strides, PaddingType padding)
{
    if (padding == PaddingType::kVALID)
        conv2D<PaddingType::kVALID>(x, k, y, shape, kernel, strides);
    else  // padding == PaddingType::kSAME
        conv2D<PaddingType::kSAME>(x, k, y, shape, kernel, strides);
}

void runConv2DGradientHost(const float* in, const float* ker, const float* out,
                           float* inG, float* kerG,
                           const std::vector<int>& shape,
                           const std::vector<int>& kernel,
                           const std::vector<int>& strides, PaddingType padding)
{
    if (padding == PaddingType::kVALID)
        conv2DGradient<PaddingType::kVALID>(in, ker, out, inG, kerG, shape,
                                            kernel, strides);
    else  // padding == PaddingType::kSAME
        conv2DGradient<PaddingType::kSAME>(in, ker, out, inG, kerG, shape,
                                           kernel, strides);
}

}  // namespace

Conv2DLayer::Conv2DLayer(ID id, const Tensor::SPtr& t,
                         const Tensor::SPtr& kernel,
                         const std::vector<int>& strides, PaddingType padding)
    : DifferentiableLayer(id, {t, kernel},
                          {createOutput(t, kernel, strides, padding)}),
      mStrides(strides),
      mPadding(padding),
      mGpuParams(t->getType(), 11)
{
    assert(t->getType() == kernel->getType());
}

Layer::TensorMap Conv2DLayer::gradients(Tensor::SPtr out, Tensor::SPtr outGrad)
{
    assert(mOutputs[0] == out);

    Tensor::SPtr in = mInputs[0].lock();
    Tensor::SPtr k = mInputs[1].lock();

    Layer::SPtr layer = createLayer<Conv2DGradientLayer>(in, k, out, outGrad,
                                                         mStrides, mPadding);

    return {{in, layer->getOutputs()[0]}, {k, layer->getOutputs()[1]}};
}

void Conv2DLayer::execute(const std::vector<float*>& inputs,
                          const std::vector<float*>& outputs,
                          const InputDict& /*inputDict*/)
{
    float* x = inputs[0];
    float* ker = inputs[1];
    float* y = outputs[0];

    TensorShape tShape = mInputs[0].lock()->getShape();
    TensorShape kShape = mInputs[1].lock()->getShape();
    std::vector<int> inShape(tShape.begin(), tShape.end());
    std::vector<int> kerShape(kShape.begin(), kShape.end());

    if (mInputs[0].lock()->getType() == MemoryType::kHOST_MEMORY)
        runConv2DHost(x, ker, y, inShape, kerShape, mStrides, mPadding);
#ifdef CUDA_AVAILABLE
    else
    {
        std::vector<int> outShape = mOutputs[0]->getShape();
        size_t size = outShape[0] * outShape[1] * outShape[2] * outShape[3];
        cuda::runConv2DDevice(x, ker, y, size, mGpuParams.getValues(),
                              mPadding);
    }
#endif
}

void Conv2DLayer::initialize()
{
    Tensor::SPtr inTensor = mInputs[0].lock();
    Tensor::SPtr kerTensor = mInputs[1].lock();

    std::vector<int> inShape = inTensor->getShape();
    std::vector<int> kShape = kerTensor->getShape();
    std::vector<int> outShape = mOutputs[0]->getShape();

    if (inTensor->getType() == MemoryType::kHOST_MEMORY)
    {
    }
#ifdef CUDA_AVAILABLE
    else
    {
        mGpuParams.allocate();
        cuda::initializeConvGpuParams(mGpuParams.getValues(), inShape.data(),
                                      kShape.data(), outShape.data(),
                                      mStrides.data());
    }
#endif
}

Conv2DLayer::~Conv2DLayer()
{
    mGpuParams.free();
}

Conv2DGradientLayer::Conv2DGradientLayer(ID id, const Tensor::SPtr& t,
                                         const Tensor::SPtr& k,
                                         const Tensor::SPtr& out,
                                         const Tensor::SPtr& outG,
                                         std::vector<int> strides,
                                         PaddingType padding)
    : Layer(id, {t, k, out, outG},
            {createTensor("", t->getShape(), t->getType()),
             createTensor("", k->getShape(), k->getType())}),
      mStrides(std::move(strides)),
      mPadding(padding),
      mGpuParams(t->getType(), 11)
{
}

void Conv2DGradientLayer::execute(const std::vector<float*>& inputs,
                                  const std::vector<float*>& outputs,
                                  const InputDict& /*inputDict*/)
{
    float* x = inputs[0];
    float* ker = inputs[1];
    float* yG = inputs[3];
    float* xG = outputs[0];
    float* kerG = outputs[1];

    TensorShape tShape = mInputs[0].lock()->getShape();
    TensorShape kShape = mInputs[1].lock()->getShape();
    std::vector<int> inShape(tShape.begin(), tShape.end());
    std::vector<int> kerShape(kShape.begin(), kShape.end());

    if (mInputs[0].lock()->getType() == MemoryType::kHOST_MEMORY)
        runConv2DGradientHost(x, ker, yG, xG, kerG, inShape, kerShape, mStrides,
                              mPadding);
#ifdef CUDA_AVAILABLE
    else
    {
        std::vector<int> inShape = mInputs[0].lock()->getShape();
        std::vector<int> kerShape = mInputs[1].lock()->getShape();
        size_t inSize = inShape[0] * inShape[1] * inShape[2] * inShape[3];
        size_t kerSize = kerShape[0] * kerShape[1] * kerShape[2] * kerShape[3];
        cuda::runConv2DGradientDevice(x, ker, yG, xG, kerG, inSize, kerSize,
                                      mGpuParams.getValues(), mPadding);
    }
#endif
}

void Conv2DGradientLayer::initialize()
{
    Tensor::SPtr inTensor = mInputs[0].lock();
    Tensor::SPtr kerTensor = mInputs[1].lock();
    Tensor::SPtr outTensor = mInputs[2].lock();

    std::vector<int> inShape = inTensor->getShape();
    std::vector<int> kShape = kerTensor->getShape();
    std::vector<int> outShape = outTensor->getShape();

    if (inTensor->getType() == MemoryType::kHOST_MEMORY)
    {
    }
#ifdef CUDA_AVAILABLE
    else
    {
        mGpuParams.allocate();
        cuda::initializeConvGpuParams(mGpuParams.getValues(), inShape.data(),
                                      kShape.data(), outShape.data(),
                                      mStrides.data());
    }
#endif
}

Conv2DGradientLayer::~Conv2DGradientLayer()
{
    mGpuParams.free();
}

}  // namespace layers

Tensor::SPtr convolution2D(const Tensor::SPtr& t, const Tensor::SPtr& kernel,
                           const std::vector<int>& strides,
                           layers::PaddingType padding)
{
    TensorShape xShape = t->getShape();
    TensorShape kShape = kernel->getShape();

    if (xShape.size() != 4)
        throw std::runtime_error("conv2D: wrong input shape");
    if (kShape.size() != 4)
        throw std::runtime_error("conv2D: wrong kernel shape");
    if (strides.empty() || strides.size() > 2)
        throw std::runtime_error("conv2D: wrong strides");

    if (xShape[1] != kShape[1])
        throw std::runtime_error("conv2D: kernel doesn\'t match tensor");

    for (int d : strides)
        if (d <= 0)
            throw std::runtime_error("conv2D: stride dims must be positive");

    Layer::SPtr layer =
        createLayer<layers::Conv2DLayer>(t, kernel, strides, padding);
    return layer->getOutputs()[0];
}

}  // namespace core

ITensorPtr conv2D(const ITensorPtr& tensor, const ITensorPtr& kernel,
                  const std::vector<int>& strides, const std::string& padding)
{
    core::layers::PaddingType p;
    if (padding == "valid" || padding == "VALID")
        p = core::layers::PaddingType::kVALID;
    else if (padding == "same" || padding == "SAME")
        p = core::layers::PaddingType::kSAME;
    else
        throw std::runtime_error(
            R"(maxPool2D: wrong padding, should be one of: "SAME", "VALID")");

    core::Tensor::SPtr t = core::castITensorPtr(tensor)->get();
    core::Tensor::SPtr k = core::castITensorPtr(kernel)->get();
    return core::makeAbstractTensor(core::convolution2D(t, k, strides, p));
}

}  // namespace graphdl
