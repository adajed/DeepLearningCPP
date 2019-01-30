#ifndef READ_CIFAR10_H_
#define READ_CIFAR10_H_

#include <string>
#include <vector>

class Cifar10Dataset
{
  public:
    Cifar10Dataset(const std::vector<std::string>& paths,
                 int batchSize);

    int getNumBatches() const;

    std::vector<std::vector<float>> getNextBatch();

    void reset();

  private:
    int mBatchSize;
    int mPos;
    std::vector<int> mIndexes;
    std::vector<std::vector<float>> mX;
    std::vector<std::vector<float>> mY;
};

#endif  // READ_CIFAR10_H_
