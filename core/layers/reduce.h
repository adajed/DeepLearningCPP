#ifndef GRAPHDL_CORE_LAYERS_REDUCE_SUM_H_
#define GRAPHDL_CORE_LAYERS_REDUCE_SUM_H_

#include "differentiableLayer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
enum class ReduceType
{
    kSUM = 0
};

class ReduceBackLayer : public DifferentiableLayer
{
  public:
    ReduceBackLayer(ID id, const Tensor::SPtr& tensor, int numAxes,
                    ReduceType reduceType);

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    int mNumAxes;
    ReduceType mReduceType;
};

class ReduceBackGradientLayer : public Layer
{
  public:
    ReduceBackGradientLayer(ID id, const Tensor::SPtr& in,
                            const Tensor::SPtr& out,
                            const Tensor::SPtr& outGrad, int numAxes,
                            ReduceType reduceType);

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    int mNumAxes;
    ReduceType mReduceType;
};

class ReduceFrontLayer : public DifferentiableLayer
{
  public:
    ReduceFrontLayer(ID id, const Tensor::SPtr& tensor, int numAxes,
                     ReduceType reduceType);

    TensorMap gradients(Tensor::SPtr out, Tensor::SPtr outGrad) override;

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    int mNumAxes;
    ReduceType mReduceType;
};

class ReduceFrontGradientLayer : public Layer
{
  public:
    ReduceFrontGradientLayer(ID id, const Tensor::SPtr& in,
                             const Tensor::SPtr& out,
                             const Tensor::SPtr& outGrad, int numAxes,
                             ReduceType reduceType);

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    int mNumAxes;
    ReduceType mReduceType;
};

#ifdef CUDA_AVAILABLE
namespace cuda
{
void runReduceBackDevice(const float* in, float* out, size_t outSize,
                         size_t reduceSize, ReduceType reduceType);

void runReduceBackGradientDevice(const float* in, const float* out,
                                 const float* outGrad, float* inGrad,
                                 size_t outSize, size_t reduceSize,
                                 ReduceType reduceType);

void runReduceFrontDevice(const float* in, float* out, size_t outSize,
                          size_t reduceSize, ReduceType reduceType);

void runReduceFrontGradientDevice(const float* in, const float* out,
                                  const float* outGrad, float* inGrad,
                                  size_t outSize, size_t reduceSize,
                                  ReduceType reduceType);

}  // namespace cuda
#endif

void runReduceBackHost(const float* in, float* out, size_t outSize,
                       size_t reduceSize, ReduceType reduceType);

void runReduceBackGradientHost(const float* in, const float* out,
                               const float* outGrad, float* inGrad,
                               size_t outSize, size_t reduceSize,
                               ReduceType reduceType);

void runReduceFrontHost(const float* in, float* out, size_t outSize,
                        size_t reduceSize, ReduceType reduceType);

void runReduceFrontGradientHost(const float* in, const float* out,
                                const float* outGrad, float* inGrad,
                                size_t outSize, size_t reduceSize,
                                ReduceType reduceType);

}  // namespace layers

Tensor::SPtr reduceBack(const Tensor::SPtr& t, int numAxes,
                        layers::ReduceType reduceType);

Tensor::SPtr reduceFront(const Tensor::SPtr& t, int numAxes,
                         layers::ReduceType reduceType);

}  // namespace core

}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_REDUCE_SUM_H_
