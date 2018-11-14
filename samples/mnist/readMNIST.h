#ifndef READ_MNIST_H_
#define READ_MNIST_H_

#include <string>
#include <vector>

class MnistDataset
{
  public:
    MnistDataset(const std::string& imagesPath, const std::string& labelsPath,
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

#endif  // READ_MNIST_H_
