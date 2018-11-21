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

  private:
    void execute(const InputDict& inputs) override;
};

Tensor::SPtr queue(const std::vector<Tensor::SPtr>& tensors);

}  // namespace layers
}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_QUEUE_H_
