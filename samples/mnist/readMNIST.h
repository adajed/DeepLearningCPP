#ifndef READ_MNIST_H_
#define READ_MNIST_H_

#include <vector>

class MnistDataset
{
  public:
    MnistDataset(int batchSize);

    int getNumBatches() const;

    std::vector<std::vector<float>> getNextBatch();

    void reset();

  private:
    int mBatchSize;
    int mPos;
    std::vector<std::vector<float>> mX;
    std::vector<std::vector<float>> mY;
};

#endif  // READ_MNIST_H_
