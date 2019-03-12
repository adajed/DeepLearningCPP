#include "graphdl.h"
#include "graphdl_ops.h"
#include "graphdl_train.h"
#include "readMNIST.h"
#include "utils.h"

#include <iostream>
#include <random>

// learning parameters
const int BATCH_SIZE = 64;  // how many samples per computation
const int NUM_EPOCHS = 1;  // # of runs over whole dataset
const int PRINT_EVERY = 100;  // after how many batches print info
const float LEARNING_RATE = 0.001;  // learning parameter to the optimizer

#define Q(x) std::string(#x)
#define QUOTE(x) Q(x)

const std::string TRAIN_IMAGES_PATH =
    QUOTE(PROJECT_DIR) + "/samples/mnist/train-images-idx3-ubyte";
const std::string TRAIN_LABELS_PATH =
    QUOTE(PROJECT_DIR) + "/samples/mnist/train-labels-idx1-ubyte";
const std::string VALID_IMAGES_PATH =
    QUOTE(PROJECT_DIR) + "/samples/mnist/t10k-images-idx3-ubyte";
const std::string VALID_LABELS_PATH =
    QUOTE(PROJECT_DIR) + "/samples/mnist/t10k-labels-idx1-ubyte";

#undef Q
#undef QUOTE

using namespace graphdl;

//! \fn buildNetwork
//! \brief Builds computation graph.
//!
ComputationalGraph buildNetwork()
{
#ifdef CUDA_AVAILABLE
    MemoryLocation loc = MemoryLocation::kDEVICE;
#else
    MemoryLocation loc = MemoryLocation::kHOST;
#endif

    SharedPtr<IInitializer> init = uniformInitializer(-1., 1., 0);

    // network inputs
    ITensorPtr X = createInput("X", {BATCH_SIZE, 28 * 28}, loc);
    ITensorPtr Y = createInput("Y", {BATCH_SIZE, 10}, loc);

    // weights
    ITensorPtr W1 = createWeights("W1", {28 * 28, 512}, init, loc);
    ITensorPtr W2 = createWeights("W2", {512, 128}, init, loc);
    ITensorPtr W3 = createWeights("W3", {128, 10}, init, loc);

    // biases
    ITensorPtr b1 = createWeights("b1", {512}, init, loc);
    ITensorPtr b2 = createWeights("b2", {128}, init, loc);
    ITensorPtr b3 = createWeights("b3", {10}, init, loc);

    ITensorPtr a1 = sigmoid(matmul(X, W1) + b1);
    ITensorPtr a2 = sigmoid(matmul(a1, W2) + b2);
    ITensorPtr a3 = sigmoid(matmul(a2, W3) + b3);

    // calculate loss
    ITensorPtr loss = neg(reduceSum(Y * log(a3) + (1. - Y) * log(1. - a3)));
    loss = loss / float(BATCH_SIZE);

    // optimize weights and biases
    ITensorPtr opt =
        train::adam(LEARNING_RATE, 0.9, 0.999, 10e-8)->optimize(loss);

    // create network
    ComputationalGraph net;
    net.inputs = {{"X", X}, {"Y", Y}};
    net.weights = {W1, W2, W3};
    net.output = a3;
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
    ComputationalGraph net = buildNetwork();
    initializeGraph();

    std::vector<float> losses;
    std::vector<int> accuracies;
    for (int e = 0; e < NUM_EPOCHS; ++e)
    {
        losses.clear();
        accuracies.clear();
        std::cout << "Epoch " << e << std::endl;
        std::cout << "Number of batches " << train_mnist.getNumBatches()
                  << std::endl;
        for (int i = 0; i < train_mnist.getNumBatches(); ++i)
        {
            auto batch = train_mnist.getNextBatch();
            auto outputs = eval({net.loss, net.output, net.optimize},
                                {{"X", batch[0]}, {"Y", batch[1]}});

            float loss = outputs[0][0];
            float acc = calcNumCorrect(batch[1], outputs[1], BATCH_SIZE);
            losses.push_back(loss);
            accuracies.push_back(acc);

            if (i % PRINT_EVERY == PRINT_EVERY - 1)
            {
                std::cout << "Step " << i << ": "
                          << "loss " << mean(losses) << ", acc "
                          << accuracy(accuracies, BATCH_SIZE) << std::endl;

                losses.clear();
                accuracies.clear();
            }
        }

        losses.clear();
        accuracies.clear();
        for (int i = 0; i < valid_mnist.getNumBatches(); ++i)
        {
            auto batch = valid_mnist.getNextBatch();
            auto outputs = eval({net.loss, net.output},
                                {{"X", batch[0]}, {"Y", batch[1]}});
            float loss = outputs[0][0];
            float acc = calcNumCorrect(batch[1], outputs[1], BATCH_SIZE);
            losses.push_back(loss);
            accuracies.push_back(acc);
        }
        std::cout << "Valid. loss " << mean(losses) << ", Valid. acc "
                  << accuracy(accuracies, BATCH_SIZE) << std::endl;
        train_mnist.reset();
        valid_mnist.reset();
    }

    return 0;
};
