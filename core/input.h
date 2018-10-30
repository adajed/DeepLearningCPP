#ifndef GRAPHDL_CORE_INPUT_H_
#define GRAPHDL_CORE_INPUT_H_

#include "graph.h"
#include "layer.h"

namespace graphdl
{
namespace core
{
class InputLayer : public Layer
{
   public:
    InputLayer(Graph::SPtr, graph, const std::string& name, const Shape& shape);

   private:
    void execute(const InputDict& inputs) override;
};

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_INPUT_H_
