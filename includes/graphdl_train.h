#ifndef GRAPHDL_TRAIN_H_
#define GRAPHDL_TRAIN_H_

#include "graphdl.h"

namespace graphdl
{
namespace train
{
class ITrainer
{
  public:
    virtual ITensorPtr optimize(const ITensorPtr& tensor) const = 0;

    virtual ~ITrainer() {}
};

using ITrainerPtr = std::unique_ptr<ITrainer>;

ITrainerPtr gradientDescent(float lr);

ITrainerPtr momentum(float lr, float m);

}  // namespace train
}  // namespace graphdl

#endif  // GRAPHDL_TRAIN_H_
