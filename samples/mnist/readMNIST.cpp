#include "readMNIST.h"
#include <assert.h>
#include <fstream>

int rev(int n)
{
    unsigned char c1 = n & 255;
    unsigned char c2 = (n >> 8) & 255;
    unsigned char c3 = (n >> 16) & 255;
    unsigned char c4 = (n >> 24) & 255;

    return (int(c1) << 24) + (int(c2) << 16) + (int(c3) << 8) + c4;
}

void parseImages(std::string path, std::vector<std::vector<float>>& db)
{
    std::ifstream file(path);
    if (file.is_open())
    {
        int mn, N, R, C;
        file.read((char*)&mn, sizeof(mn));
        mn = rev(mn);
        file.read((char*)&N, sizeof(N));
        N = rev(N);
        file.read((char*)&R, sizeof(R));
        R = rev(R);
        file.read((char*)&C, sizeof(C));
        C = rev(C);

        unsigned char temp;
        for (int n = 0; n < N; ++n)
        {
            db.push_back({});
            for (int r = 0; r < R; ++r)
            {
                for (int c = 0; c < C; ++c)
                {
                    file.read((char*)&temp, sizeof(temp));
                    db.back().push_back(float(temp) / 255.);
                }
            }
        }
    }
}

void parseLabels(std::string path, std::vector<std::vector<float>>& db)
{
    std::ifstream file(path);
    if (file.is_open())
    {
        int mn, N;
        file.read((char*)&mn, sizeof(mn));
        mn = rev(mn);
        file.read((char*)&N, sizeof(N));
        N = rev(N);

        unsigned char temp;
        for (int n = 0; n < N; ++n)
        {
            db.push_back({});
            file.read((char*)&temp, sizeof(temp));
            for (int i = 0; i < 10; ++i)
            {
                float v = (i == int(temp)) ? 1. : 0.;
                db.back().push_back(v);
            }
        }
    }
}

MnistDataset::MnistDataset(int batchSize)
    : mBatchSize(batchSize), mPos(0), mX(), mY()
{
    const std::string IMAGES_PATH =
        "/home/adam/Projects/DLL/samples/mnist/train-images-idx3-ubyte";
    const std::string LABELS_PATH =
        "/home/adam/Projects/DLL/samples/mnist/train-labels-idx1-ubyte";

    parseImages(IMAGES_PATH, mX);
    parseLabels(LABELS_PATH, mY);
}

int MnistDataset::getNumBatches() const { return mX.size() / mBatchSize; }

std::vector<std::vector<float>> MnistDataset::getNextBatch()
{
    std::vector<float> batchX, batchY;
    for (int n = 0; n < mBatchSize; ++n)
    {
        for (float f : mX[mPos + n]) batchX.push_back(f);
        for (float f : mY[mPos + n]) batchY.push_back(f);
    }

    mPos += mBatchSize;
    return {batchX, batchY};
}

void MnistDataset::reset() { mPos = 0; }
