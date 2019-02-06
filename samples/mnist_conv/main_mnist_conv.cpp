#include "graphdl.h"
#include "graphdl_ops.h"
#include "graphdl_train.h"
#include "readMNIST.h"

#include <iostream>
#include <numeric>
#include <random>

#ifdef CUDA_AVAILABLE
const int BATCH_SIZE = 64;
#else
const int BATCH_SIZE = 16;
#endif
const int NUM_EPOCHS = 1;
const int PRINT_EVERY = 100;
const float LEARNING_RATE = 0.001;

const std::string TRAIN_IMAGES_PATH =
    "/home/adam/Projects/DLL/samples/mnist_conv/train-images-idx3-ubyte";
const std::string TRAIN_LABELS_PATH =
    "/home/adam/Projects/DLL/samples/mnist_conv/train-labels-idx1-ubyte";
const std::string VALID_IMAGES_PATH =
    "/home/adam/Projects/DLL/samples/mnist_conv/t10k-images-idx3-ubyte";
const std::string VALID_LABELS_PATH =
    "/home/adam/Projects/DLL/samples/mnist_conv/t10k-labels-idx1-ubyte";

using namespace graphdl;

struct Network
{
    std::map<std::string, ITensorPtr> inputs;
    std::vector<ITensorPtr> weights;
    ITensorPtr output, loss, optimize;
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

float mean(const std::vector<float>& vec)
{
    float m = std::accumulate(vec.begin(), vec.end(), 0.);
    return m / float(vec.size());
}

float meanAcc(const std::vector<int>& vec)
{
    int sum = std::accumulate(vec.begin(), vec.end(), 0);
    return float(sum) / float(vec.size() * BATCH_SIZE);
}

Network buildNetwork()
{
#ifdef CUDA_AVAILABLE
    MemoryLocation loc = MemoryLocation::kDEVICE;
#else
    MemoryLocation loc = MemoryLocation::kHOST;
#endif

    SharedPtr<IInitializer> init = uniformInitializer(-1., 1., 0);
    SharedPtr<IInitializer> initW1 = normalInitializer(0., 1. / 5., 0);
    SharedPtr<IInitializer> initW2 = normalInitializer(0., 1. / 20., 0);
    SharedPtr<IInitializer> initW3 = normalInitializer(0., 1. / 48., 0);
    ITensorPtr X = createInput("X", {BATCH_SIZE, 1, 28, 28}, loc);
    ITensorPtr Y = createInput("Y", {BATCH_SIZE, 10}, loc);

    ITensorPtr W1 = createWeights("W1", {4, 1, 3, 3}, initW1, loc);
    ITensorPtr W2 = createWeights("W2", {16, 4, 3, 3}, initW2, loc);
    ITensorPtr W3 = createWeights("W3", {32, 16, 3, 3}, initW3, loc);
    ITensorPtr W4 = createWeights("W4", {32 * 7 * 7, 128}, init, loc);
    ITensorPtr W5 = createWeights("W5", {128, 10}, init, loc);
    ITensorPtr b1 = createWeights("b1", {128}, init, loc);
    ITensorPtr b2 = createWeights("b2", {10}, init, loc);

    ITensorPtr a;
    a = conv2D(X, W1, {1, 1}, "SAME");
    a = maxPool2D(a, {2, 2}, {2, 2}, "SAME");
    a = relu(a);
    a = conv2D(a, W2, {1, 1}, "SAME");
    a = maxPool2D(a, {2, 2}, {2, 2}, "SAME");
    a = relu(a);
    a = conv2D(a, W3, {1, 1}, "SAME");
    a = reshape(a, {BATCH_SIZE, 32 * 7 * 7});
    a = relu(matmul(a, W4) + b1);
    a = matmul(a, W5) + b2;
    a = softmax(a, 1);

    ITensorPtr loss = neg(reduceSum(Y * log(a))) / float(BATCH_SIZE);

    ITensorPtr opt =
        train::adam(LEARNING_RATE, 0.9, 0.999, 10e-8)->optimize(loss);

    Network net;
    net.inputs = {{"X", X}, {"Y", Y}};
    net.weights = {W1, W2, W3};
    net.output = a;
    net.loss = loss;
    net.optimize = opt;
    return net;
}

int main()
{
    std::cout << "Reading MNIST dataset..." << std::endl;
    MnistDataset train_mnist(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, BATCH_SIZE);
    MnistDataset valid_mnist(VALID_IMAGES_PATH, VALID_LABELS_PATH, BATCH_SIZE);
    std::cout << "Building network..." << std::endl;
    Network net = buildNetwork();
    initializeGraph();

    std::vector<float> losses;
    std::vector<int> accs;
    std::cout << "Number of epochs: " << NUM_EPOCHS << std::endl;
    for (int e = 0; e < NUM_EPOCHS; ++e)
    {
        losses.clear();
        accs.clear();
        std::cout << "Epoch " << e << std::endl;
        std::cout << "Number of batches " << train_mnist.getNumBatches()
                  << std::endl;
        for (int i = 0; i < train_mnist.getNumBatches(); ++i)
        {
            auto batch = train_mnist.getNextBatch();
            auto outputs = eval({net.loss, net.output, net.optimize},
                                {{"X", batch[0]}, {"Y", batch[1]}});
            losses.push_back(outputs[0][0]);
            accs.push_back(numCorrect(batch[1], outputs[1]));
            if (i % PRINT_EVERY == PRINT_EVERY - 1)
            {
                std::cout << "Step " << i << ": "
                          << "loss " << mean(losses) << ", acc "
                          << meanAcc(accs) << std::endl;

                losses.clear();
                accs.clear();
            }
        }

        losses.clear();
        accs.clear();
        for (int i = 0; i < valid_mnist.getNumBatches(); ++i)
        {
            auto batch = valid_mnist.getNextBatch();
            auto outputs = eval({net.loss, net.output},
                                {{"X", batch[0]}, {"Y", batch[1]}});
            losses.push_back(outputs[0][0]);
            accs.push_back(numCorrect(batch[1], outputs[1]));
        }
        std::cout << "Valid. loss " << mean(losses) << ", Valid. acc "
                  << meanAcc(accs) << std::endl;
        train_mnist.reset();
        valid_mnist.reset();
    }

    return 0;
};