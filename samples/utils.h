#ifndef SAMPLES_UTILS_H_
#define SAMPLES_UTILS_H_

#include <numeric>

using namespace graphdl;

struct ComputationalGraph
{
    std::map<std::string, ITensorPtr> inputs;
    std::vector<ITensorPtr> weights;
    ITensorPtr output;
    ITensorPtr loss;
    ITensorPtr optimize;
};

int calcNumCorrect(const HostTensor& y, const HostTensor& pred, int n)
{
    int count = 0;
    for (int b = 0; b < n; ++b)
    {
        int bestY = 0, bestPred = 0;
        float maxY = y[10 * b], maxPred = pred[10 * b];
        for (int i = 1; i < 10; ++i)
        {
            if (y[10 * b + i] > maxY)
            {
                bestY = i;
                maxY = y[10 * b + i];
            }
            if (pred[10 * b + i] > maxPred)
            {
                bestPred = i;
                maxPred = pred[10 * b + i];
            }
        }

        if (bestY == bestPred) count++;
    }

    return count;
}

float mean(const std::vector<float>& v)
{
    float m = std::accumulate(v.begin(), v.end(), 0.);
    return m / float(v.size());
}

float accuracy(const std::vector<int>& v, int n)
{
    int sum = std::accumulate(v.begin(), v.end(), 0);
    return float(sum) / float(v.size() * n);
}

#endif  // SAMPLES_UTILS_H_
