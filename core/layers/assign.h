#ifndef GRAPHDL_CORE_LAYERS_ASSIGN_H_
#define GRAPHDL_CORE_LAYERS_ASSIGN_H_

#include "layer.h"

namespace graphdl
{
namespace core
{
namespace layers
{
class AssignLayer : public Layer
{
  public:
    AssignLayer(ID id, Tensor::SPtr dest, Tensor::SPtr src);

  private:
    void execute(const InputDict& inputs) override;

    Tensor::WeakPtr mDest;
};

}  // namespace layers

Tensor::SPtr assign(Tensor::SPtr dest, Tensor::SPtr src);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_LAYERS_ASSIGN_H_
