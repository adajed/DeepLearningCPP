#ifndef GRAPHDL_CORE_LAYERS_QUEUE_H_
#define GRAPHDL_CORE_LAYERS_QUEUE_H_

#include "layer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
class QueueLayer : public Layer
{
  public:
    QueueLayer(ID id, const std::vector<Tensor::SPtr>& ops);

    void eval(const InputDict& inputDict) override;

  private:
    void execute(const std::vector<float*>& inputs,
                 const std::vector<float*>& outputs,
                 const InputDict& inputDict) override;

    std::vector<Tensor::WeakPtr> mOps;
};

}  // namespace layers

Tensor::SPtr queue(const std::vector<Tensor::SPtr>& ops);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_QUEUE_H_
