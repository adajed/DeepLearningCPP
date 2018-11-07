#ifndef GRAPHDL_CORE_INPUT_H_
#define GRAPHDL_CORE_INPUT_H_

#include "layer.h"

namespace graphdl
{
namespace core
{
namespace cuda
{
extern "C" void copyInput(float* dest, float* src, size_t count);
}

class InputLayer : public Layer
{
  public:
    InputLayer(ID id, const std::string& name, const Shape& shape,
               MemoryType type);

  private:
    void execute(const InputDict& inputs) override;
};

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_INPUT_H_
