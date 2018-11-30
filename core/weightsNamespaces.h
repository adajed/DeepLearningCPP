#ifndef GRAPHDL_CORE_WEIGHTS_NAMESPACE_H_
#define GRAPHDL_CORE_WEIGHTS_NAMESPACE_H_

#include "layer.h"

#include <map>
#include <string>
#include <vector>

namespace graphdl
{
namespace core
{
extern const std::string GRAPH_WEIGHTS_NAMESPACE;
extern const std::string TRAIN_WEIGHTS_NAMESPACE;

class WeightsNamespaces : public std::map<std::string, std::vector<Layer::SPtr>>
{
  public:
    //! this constructor creates default namespaces
    WeightsNamespaces()
        : std::map<std::string, std::vector<Layer::SPtr>>(
              {{GRAPH_WEIGHTS_NAMESPACE, {}}, {TRAIN_WEIGHTS_NAMESPACE, {}}})
    {
    }

    bool contains(const Layer::SPtr& layer) const;
};

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_WEIGHTS_NAMESPACE_H_
