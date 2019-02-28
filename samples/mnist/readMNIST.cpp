#include "readMNIST.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

bool fileExists(const std::string& path)
{
    std::ifstream f(path.c_str());
    if (!f.good())
    {
        std::cout << "File \"" << path
                  << "\" does not exists, please download it." << std::endl;
        return false;
    }

    return true;
}

unsigned rev(unsigned n)
{
    unsigned char c1 = n & 255;
    unsigned char c2 = (n >> 8) & 255;
    unsigned char c3 = (n >> 16) & 255;
    unsigned char c4 = (n >> 24) & 255;

    return (unsigned(c1) << 24) + (unsigned(c2) << 16) + (unsigned(c3) << 8) +
           c4;
}

void parseImages(const std::string& path, std::vector<std::vector<float>>& db)
{
    std::ifstream file(path);
    if (file.is_open())
    {
        unsigned mn, N, R, C;
        file.read(reinterpret_cast<char*>(&mn), sizeof(mn));
        mn = rev(mn);
        file.read(reinterpret_cast<char*>(&N), sizeof(N));
        N = rev(N);
        file.read(reinterpret_cast<char*>(&R), sizeof(R));
        R = rev(R);
        file.read(reinterpret_cast<char*>(&C), sizeof(C));
        C = rev(C);

        unsigned char temp;
        for (unsigned n = 0; n < N; ++n)
        {
            db.emplace_back();
            for (unsigned r = 0; r < R; ++r)
            {
                for (unsigned c = 0; c < C; ++c)
                {
                    file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
                    db.back().push_back(float(temp) / 255.);
                }
            }
        }
    }
}

void parseLabels(const std::string& path, std::vector<std::vector<float>>& db)
{
    std::ifstream file(path);
    if (file.is_open())
    {
        unsigned mn, N;
        file.read(reinterpret_cast<char*>(&mn), sizeof(mn));
        mn = rev(mn);
        file.read(reinterpret_cast<char*>(&N), sizeof(N));
        N = rev(N);

        unsigned char temp;
        for (unsigned n = 0; n < N; ++n)
        {
            db.emplace_back();
            file.read(reinterpret_cast<char*>(&temp), sizeof(temp));
            for (unsigned i = 0; i < 10; ++i)
            {
                float v = (i == unsigned(temp)) ? 1. : 0.;
                db.back().push_back(v);
            }
        }
    }
}

MnistDataset::MnistDataset(const std::string& imagesPath,
                           const std::string& labelsPath, int batchSize)
    : mBatchSize(batchSize), mPos(0)
{
    assert(fileExists(imagesPath));
    assert(fileExists(labelsPath));

    parseImages(imagesPath, mX);
    parseLabels(labelsPath, mY);
    mIndexes = std::vector<int>(mX.size());
    std::iota(mIndexes.begin(), mIndexes.end(), 0);
    std::shuffle(mIndexes.begin(), mIndexes.end(),
                 std::mt19937(std::random_device()()));
}

int MnistDataset::getNumBatches() const
{
    return mX.size() / mBatchSize;
}

std::vector<std::vector<float>> MnistDataset::getNextBatch()
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

void MnistDataset::reset()
{
    mPos = 0;
    std::shuffle(mIndexes.begin(), mIndexes.end(),
                 std::mt19937(std::random_device()()));
}
