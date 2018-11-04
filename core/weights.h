#ifndef GRAPHDL_CORE_WEIGHTS_H_
#define GRAPHDL_CORE_WEIGHTS_H_

#include "layer.h"

namespace graphdl
{
namespace core
{
namespace cuda
{
extern "C" void initWeights(float* memory, size_t count);
}

class WeightsLayer : public Layer
{
   public:
    WeightsLayer(ID id, const std::string& name, const Shape& shape,
                 MemoryType type);

    void initialize() override;

   private:
    void execute(const InputDict& inputs) override;
};

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_WEIGHTS_H_
