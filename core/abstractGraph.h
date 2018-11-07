#ifndef GRAPHDL_CORE_ABSTRACT_GRAPH_H_
#define GRAPHDL_CORE_ABSTRACT_GRAPH_H_

#include "graph.h"
#include "graphdl.h"

namespace graphdl
{
namespace core
{
//! \class AbstractGraph
//! \brief Implementation of IGraph.
//!
class AbstractGraph : public IGraph
{
  public:
    using Ptr = std::shared_ptr<AbstractGraph>;

    AbstractGraph(Graph::SPtr graph);

    std::string getName() const override;

    void setName(const std::string& name) override;

    std::map<std::string, ITensorPtr> getInputs() const override;

    std::map<std::string, ITensorPtr> getWeights() const override;

    Graph::SPtr get() const;

  private:
    Graph::SPtr mGraph;
};

AbstractGraph::Ptr makeAbstractGraph(Graph::SPtr graph);

AbstractGraph::Ptr castIGraphPtr(const IGraphPtr& igraph);

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_ABSTRACT_GRAPH_H_
