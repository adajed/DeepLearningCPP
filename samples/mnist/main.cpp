#include "graphdl.h"
#include "graphdl_ops.h"
#include "readMNIST.h"

#include <iostream>
#include <random>

const int BATCH_SIZE = 64;
const int NUM_EPOCHS = 1;
const int PRINT_EVERY = 100;

using namespace graphdl;

struct Network
{
    std::map<std::string, ITensorPtr> inputs;
    std::vector<ITensorPtr> weights;
    std::vector<ITensorPtr> modifiers;
    ITensorPtr output, loss;
};

int numCorrect(const HostTensor& y, const HostTensor& pred)
{
    int cnt = 0;
    for (int b = 0; b < BATCH_SIZE; ++b)
    {
        int bestY = 0, bestPred = 0;
        float maxY = y[10 * b], maxPred = pred[10 * b];
        for (int i = 0; i < 10; ++i)
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

        if (bestY == bestPred) cnt++;
    }

    return cnt;
}

Network buildNetwork()
{
#ifdef CUDA_AVAILABLE
    MemoryLocation loc = MemoryLocation::kDEVICE;
#else
    MemoryLocation loc = MemoryLocation::kHOST;
#endif
    ITensorPtr X = createInput("X", {BATCH_SIZE, 28 * 28}, loc);
    ITensorPtr Y = createInput("Y", {BATCH_SIZE, 10}, loc);

    ITensorPtr W1 = createWeights("W1", {28 * 28, 512}, loc);
    ITensorPtr W2 = createWeights("W2", {512, 128}, loc);
    ITensorPtr W3 = createWeights("W3", {128, 10}, loc);

    ITensorPtr a1 = sigmoid(matmul(X, W1));
    ITensorPtr a2 = sigmoid(matmul(a1, W2));
    ITensorPtr a3 = sigmoid(matmul(a2, W3));

    ITensorPtr ones = constant(1., Y->getShape(), loc);
    ITensorPtr loss = neg(reduceSum(Y * log(a3) + (ones - Y) * log(ones - a3)));
    loss = loss / constant(BATCH_SIZE, {}, loc);

    std::map<ITensorPtr, ITensorPtr> grads = gradients(loss);
    std::vector<ITensorPtr> modifiers;
    for (auto pair : grads)
    {
        ITensorPtr s = constant(0.1, pair.first->getShape(), loc);
        modifiers.push_back(assign(pair.first, pair.first - s * pair.second));
    }

    Network net;
    net.inputs = {{"X", X}, {"Y", Y}};
    net.weights = {W1, W2, W3};
    net.output = a3;
    net.loss = loss;
    net.modifiers = modifiers;
    return net;
}

int main()
{
    std::cout << "Reading MNIST dataset..." << std::endl;
    MnistDataset mnist(BATCH_SIZE);
    std::cout << "Building network..." << std::endl;
    Network net = buildNetwork();
    initializeGraph();

    std::vector<float> losses;
    std::vector<int> accs;
    for (int e = 0; e < NUM_EPOCHS; ++e)
    {
        losses.clear();
        accs.clear();
        std::cout << "Epoch " << e << std::endl;
        std::cout << "Number of batches " << mnist.getNumBatches() << std::endl;
        for (int i = 0; i < mnist.getNumBatches(); ++i)
        {
            auto batch = mnist.getNextBatch();
            auto outputs = eval({net.loss, net.output},
                                {{"X", batch[0]}, {"Y", batch[1]}});
            (void)eval(net.modifiers, {{"X", batch[0]}, {"Y", batch[1]}});
            losses.push_back(outputs[0][0]);
            accs.push_back(numCorrect(batch[1], outputs[1]));
            if (i % PRINT_EVERY == PRINT_EVERY - 1)
            {
                float mean = 0.;
                for (float f : losses) mean += f;
                mean /= float(losses.size());

                int acc = 0;
                for (int i : accs) acc += i;
                std::cout << "Loss " << mean << ", acc "
                          << float(acc) / float(PRINT_EVERY * BATCH_SIZE)
                          << std::endl;

                losses.clear();
                accs.clear();
            }
        }
        mnist.reset();
    }

    return 0;
};
