#include "weightsNamespaces.h"

namespace graphdl
{
namespace core
{
const std::string GRAPH_WEIGHTS_NAMESPACE = "GRAPH_WEIGHTS_NAMESPACE";
const std::string TRAIN_WEIGHTS_NAMESPACE = "TRAIN_WEIGHTS_NAMESPACE";

bool WeightsNamespaces::contains(const Layer::SPtr& layer) const
{
    for (const auto& m : *this)
        for (const auto& l : m.second)
            if (l == layer) return true;

    return false;
}

}  // namespace core
}  // namespace graphdl
