#ifndef GRAPHDL_CORE_LAYERS_GROUP_H_
#define GRAPHDL_CORE_LAYERS_GROUP_H_

#include "layer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
class GroupLayer : public Layer
{
  public:
    GroupLayer(ID id, const std::vector<Tensor::SPtr>& tensors);

  private:
    void execute(const InputDict& inputs) override;
};

}  // namespace layers

Tensor::SPtr group(const std::vector<Tensor::SPtr>& tensors);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_GROUP_H_
