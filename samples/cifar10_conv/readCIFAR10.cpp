#include "readCIFAR10.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random> 

void parse(const std::string& path, std::vector<std::vector<float>>& xs,
           std::vector<std::vector<float>>& ys)
{
    std::ifstream file(path);
    for (int n = 0; n < 10000; ++n)
    {
        std::vector<float> x(3072);
        std::vector<float> y(10);
        char byte;

        file.read(&byte, sizeof(char));
        for (int i = 0; i < 10; ++i)
            y[i] = i == int(byte) ? 1. : 0.;

        for (int i = 0; i < 3072; ++i)
        {
            file.read(&byte, sizeof(char));
            x[i] = float(byte) / 127. - 1.;
        }

        ys.push_back(y);
        xs.push_back(x);
    }
}

Cifar10Dataset::Cifar10Dataset(const std::vector<std::string>& paths,
                               int batchSize)
    : mBatchSize(batchSize), mPos(0)
{
    for (const auto& path : paths)
        parse(path, mX, mY);
    mIndexes = std::vector<int>(mX.size());
    std::iota(mIndexes.begin(), mIndexes.end(), 0);
    std::shuffle(mIndexes.begin(), mIndexes.end(),
                 std::mt19937(std::random_device()()));
}

int Cifar10Dataset::getNumBatches() const
{
    return mX.size() / mBatchSize;
}

std::vector<std::vector<float>> Cifar10Dataset::getNextBatch()
{
    std::vector<float> batchX, batchY;
    for (int n = 0; n < mBatchSize; ++n)
    {
        int i = mIndexes[mPos + n];
        for (float f : mX[i]) batchX.push_back(f);
        for (float f : mY[i]) batchY.push_back(f);
    }

    mPos += mBatchSize;
    return {batchX, batchY};
}

void Cifar10Dataset::reset()
{
    mPos = 0;
    std::shuffle(mIndexes.begin(), mIndexes.end(),
                 std::mt19937(std::random_device()()));
}
