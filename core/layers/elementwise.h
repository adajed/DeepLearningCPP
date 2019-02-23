#ifndef GRAPHDL_CORE_LAYERS_ELEMENTWISE_H_
#define GRAPHDL_CORE_LAYERS_ELEMENTWISE_H_

#include "differentiableLayer.h"

#include <functional>

namespace graphdl
{
namespace core
{
namespace layers
{
enum class Elementwise : int
{
    kADD = 0,
    kSUB = 1,
    kMUL = 2,
    kDIV = 3
};

using ElementwiseFun = std::function<float(float, float)>;

class ElementwiseBackLayer : public DifferentiableLayer
{
  public:
    ElementwiseBackLayer(ID id, const Tensor::SPtr& t1, const Tensor::SPtr& t2,
                     Elementwise op);

    TensorMap gradients(Tensor::SPtr output, Tensor::SPtr outputGrad) override;

  private:
    void execute(const InputDict& inputs) override;

    Elementwise mOp;
    ElementwiseFun mFun;
};

class ElementwiseBackGradientLayer : public Layer
{
  public:
    ElementwiseBackGradientLayer(ID id, const Tensor::SPtr& t1,
                             const Tensor::SPtr& t2, Tensor::SPtr out,
                             Tensor::SPtr outGrad, Elementwise op);

  private:
    void execute(const InputDict&) override;

    Elementwise mOp;
    ElementwiseFun mFun1, mFun2;
};

class ElementwiseFrontLayer : public DifferentiableLayer
{
  public:
    ElementwiseFrontLayer(ID id, const Tensor::SPtr& t1, const Tensor::SPtr& t2,
                     Elementwise op);

    TensorMap gradients(Tensor::SPtr output, Tensor::SPtr outputGrad) override;

  private:
    void execute(const InputDict& inputs) override;

    Elementwise mOp;
    ElementwiseFun mFun;
};

class ElementwiseFrontGradientLayer : public Layer
{
  public:
    ElementwiseFrontGradientLayer(ID id, const Tensor::SPtr& t1,
                             const Tensor::SPtr& t2, Tensor::SPtr out,
                             Tensor::SPtr outGrad, Elementwise op);

  private:
    static std::vector<Tensor::SPtr> createOutputs(Tensor::SPtr, Tensor::SPtr);

    void execute(const InputDict&) override;

    Elementwise mOp;
    ElementwiseFun mFun1, mFun2;
};

#ifdef CUDA_AVAILABLE
namespace cuda
{
void runElementwiseBackDevice(const float* x1, size_t size1,
                          const float* x2, size_t size2,
                          float* y, Elementwise op);

void runElementwiseBackGradientDevice(const float* x1, size_t size1,
                                  const float* x2, size_t size2,
                                  const float* yGrad, float* x1Grad,
                                  float* x2Grad, Elementwise op);

void runElementwiseFrontDevice(const float* x1, size_t size1,
                               const float* x2, size_t size2,
                               float* y, Elementwise op);

void runElementwiseFrontGradientDevice(const float* x1, size_t size1,
                                       const float* x2, size_t size2,
                                       const float* yGrad, float* x1Grad,
                                       float* x2Grad, Elementwise op);

}  // namespace cuda
#endif

void runElementwiseBackHost(const float* x1, size_t size1,
                        const float* x2, size_t size2,
                        float* y, Elementwise op);

void runElementwiseBackGradientHost(const float* x1, size_t size1,
                                const float* x2, size_t size2,
                                const float* yG, float* x1G,
                                float* x2G, Elementwise op);

void runElementwiseFrontHost(const float* x1, size_t size1,
                             const float* x2, size_t size2,
                             float* y, Elementwise op);

void runElementwiseFrontGradientHost(const float* x1, size_t size1,
                                     const float* x2, size_t size2,
                                     const float* yG, float* x1G,
                                     float* x2G, Elementwise op);

}  // namespace layers

Tensor::SPtr elementwiseBack(const Tensor::SPtr& t1, const Tensor::SPtr& t2,
                             layers::Elementwise op);

Tensor::SPtr elementwiseFront(const Tensor::SPtr& t1, const Tensor::SPtr& t2,
                              layers::Elementwise op);

Tensor::SPtr elementwise(const Tensor::SPtr& t1, const Tensor::SPtr& t2,
                         layers::Elementwise op);

Tensor::SPtr add(const Tensor::SPtr&, const Tensor::SPtr&);
Tensor::SPtr add(float, const Tensor::SPtr&);
Tensor::SPtr add(const Tensor::SPtr&, float);
Tensor::SPtr operator+(const Tensor::SPtr&, const Tensor::SPtr&);
Tensor::SPtr operator+(float, const Tensor::SPtr&);
Tensor::SPtr operator+(const Tensor::SPtr&, float);

Tensor::SPtr sub(const Tensor::SPtr&, const Tensor::SPtr&);
Tensor::SPtr sub(float, const Tensor::SPtr&);
Tensor::SPtr sub(const Tensor::SPtr&, float);
Tensor::SPtr operator-(const Tensor::SPtr&, const Tensor::SPtr&);
Tensor::SPtr operator-(float, const Tensor::SPtr&);
Tensor::SPtr operator-(const Tensor::SPtr&, float);

Tensor::SPtr mul(const Tensor::SPtr&, const Tensor::SPtr&);
Tensor::SPtr mul(float, const Tensor::SPtr&);
Tensor::SPtr mul(const Tensor::SPtr&, float);
Tensor::SPtr operator*(const Tensor::SPtr&, const Tensor::SPtr&);
Tensor::SPtr operator*(float, const Tensor::SPtr&);
Tensor::SPtr operator*(const Tensor::SPtr&, float);

Tensor::SPtr div(const Tensor::SPtr&, const Tensor::SPtr&);
Tensor::SPtr div(float, const Tensor::SPtr&);
Tensor::SPtr div(const Tensor::SPtr&, float);
Tensor::SPtr operator/(const Tensor::SPtr&, const Tensor::SPtr&);
Tensor::SPtr operator/(float, const Tensor::SPtr&);
Tensor::SPtr operator/(const Tensor::SPtr&, float);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_ELEMENTWISE_H_
