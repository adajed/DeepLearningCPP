#ifndef GRAPHDL_CORE_WEIGHTS_H_
#define GRAPHDL_CORE_WEIGHTS_H_

#include <random>

#include "layer.h"

namespace graphdl
{
namespace core
{
class WeightsLayer : public Layer
{
   public:
    WeightsLayer(Graph::SPtr graph, const std::string& name, const Shape& shape);

    void initialize() override;

   private:
    void execute(const InputDict& inputs) override;
};

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_WEIGHTS_H_
