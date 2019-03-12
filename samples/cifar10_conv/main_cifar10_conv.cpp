#include "graphdl.h"
#include "graphdl_ops.h"
#include "graphdl_train.h"
#include "readCIFAR10.h"
#include "utils.h"

#include <iostream>
#include <random>

const int BATCH_SIZE = 64;  // how many samples per computation
const int NUM_EPOCHS = 1;  // # of runs over whole dataset
const int PRINT_EVERY = 100;  // after how many batches print info
const float LEARNING_RATE = 0.001;  // learning parameter to the optimizer

#define Q(x) std::string(#x)
#define QUOTE(x) Q(x)

const std::vector<std::string> TRAIN_PATHS = {
    QUOTE(PROJECT_DIR) + "/samples/cifar10_conv/data_batch_1.bin",
    QUOTE(PROJECT_DIR) + "/samples/cifar10_conv/data_batch_2.bin",
    QUOTE(PROJECT_DIR) + "/samples/cifar10_conv/data_batch_3.bin",
    QUOTE(PROJECT_DIR) + "/samples/cifar10_conv/data_batch_4.bin",
    QUOTE(PROJECT_DIR) + "/samples/cifar10_conv/data_batch_5.bin",
};
const std::vector<std::string> VALID_PATHS = {
    QUOTE(PROJECT_DIR) + "/samples/cifar10_conv/test_batch.bin"};

#undef Q
#undef QUOTE

using namespace graphdl;

//! \fn conv2DAndMaxPool2D
//! \brief Helper function that does conv2D -> pool2D -> relu.
//!
ITensorPtr conv2DAndMaxPool2D(const ITensorPtr& x, const ITensorPtr& k)
{
    ITensorPtr a = conv2D(x, k, {1, 1}, "SAME");
    return relu(maxPool2D(a, {2, 2}, {2, 2}, "SAME"));
}

ComputationalGraph buildNetwork()
{
#ifdef CUDA_AVAILABLE
    MemoryLocation loc = MemoryLocation::kDEVICE;
#else
    MemoryLocation loc = MemoryLocation::kHOST;
#endif

    // initializers
    SharedPtr<IInitializer> init = uniformInitializer(-1., 1., 0);
    SharedPtr<IInitializer> initK1 = normalInitializer(0., 1. / 11., 0);
    SharedPtr<IInitializer> initK2 = normalInitializer(0., 1. / 24., 0);
    SharedPtr<IInitializer> initK3 = normalInitializer(0., 1. / 48., 0);
    SharedPtr<IInitializer> initK4 = normalInitializer(0., 1. / 96., 0);

    // inputs
    ITensorPtr X = createInput("X", {BATCH_SIZE, 3, 32, 32}, loc);
    ITensorPtr Y = createInput("Y", {BATCH_SIZE, 10}, loc);

    // convolution kernels
    ITensorPtr K1 = createWeights("K1", {8, 3, 3, 3}, initK1, loc);
    ITensorPtr K2 = createWeights("K2", {16, 8, 3, 3}, initK2, loc);
    ITensorPtr K3 = createWeights("K3", {32, 16, 3, 3}, initK3, loc);
    ITensorPtr K4 = createWeights("K4", {64, 32, 3, 3}, initK4, loc);

    // weights
    ITensorPtr W1 = createWeights("W1", {64 * 4 * 4, 128}, init, loc);
    ITensorPtr W2 = createWeights("W2", {128, 10}, init, loc);
    ITensorPtr b1 = createWeights("b1", {128}, init, loc);
    ITensorPtr b2 = createWeights("b2", {10}, init, loc);

    ITensorPtr a = conv2DAndMaxPool2D(X, K1);
    a = conv2DAndMaxPool2D(a, K2);
    a = conv2DAndMaxPool2D(a, K3);
    a = conv2D(a, K4, {1, 1}, "SAME");
    a = reshape(a, {BATCH_SIZE, 64 * 4 * 4});
    a = relu(matmul(a, W1) + b1);
    a = matmul(a, W2) + b2;
    a = softmax(a, 1);

    ITensorPtr loss = neg(reduceSum(Y * log(a))) / float(BATCH_SIZE);

    ITensorPtr opt =
        train::adam(LEARNING_RATE, 0.9, 0.999, 10e-8)->optimize(loss);

    ComputationalGraph net;
    net.inputs = {{"X", X}, {"Y", Y}};
    net.weights = {K1, K2, K3, K4, W1, W2, b1, b2};
    net.output = a;
    net.loss = loss;
    net.optimize = opt;
    return net;
}

int main()
{
    std::cout << "Reading CIFAR10 dataset..." << std::endl;
    Cifar10Dataset train_cifar10(TRAIN_PATHS, BATCH_SIZE);
    Cifar10Dataset valid_cifar10(VALID_PATHS, BATCH_SIZE);
    std::cout << "Building network..." << std::endl;
    ComputationalGraph net = buildNetwork();
    initializeGraph();

    std::vector<float> losses;
    std::vector<int> accs;
    std::cout << "Number of epochs: " << NUM_EPOCHS << std::endl;
    for (int e = 0; e < NUM_EPOCHS; ++e)
    {
        losses.clear();
        accs.clear();
        std::cout << "Epoch " << e << std::endl;
        std::cout << "Number of batches " << train_cifar10.getNumBatches()
                  << std::endl;
        for (int i = 0; i < train_cifar10.getNumBatches(); ++i)
        {
            auto batch = train_cifar10.getNextBatch();
            auto outputs = eval({net.loss, net.output, net.optimize},
                                {{"X", batch[0]}, {"Y", batch[1]}});
            losses.push_back(outputs[0][0]);
            accs.push_back(calcNumCorrect(batch[1], outputs[1], BATCH_SIZE));
            if (i % PRINT_EVERY == PRINT_EVERY - 1)
            {
                std::cout << "Step " << i << ": "
                          << "loss " << mean(losses) << ", acc "
                          << accuracy(accs, BATCH_SIZE) << std::endl;

                losses.clear();
                accs.clear();
            }
        }

        losses.clear();
        accs.clear();
        for (int i = 0; i < valid_cifar10.getNumBatches(); ++i)
        {
            auto batch = valid_cifar10.getNextBatch();
            auto outputs = eval({net.loss, net.output},
                                {{"X", batch[0]}, {"Y", batch[1]}});
            losses.push_back(outputs[0][0]);
            accs.push_back(calcNumCorrect(batch[1], outputs[1], BATCH_SIZE));
        }
        std::cout << "Valid. loss " << mean(losses) << ", Valid. acc "
                  << accuracy(accs, BATCH_SIZE) << std::endl;
        train_cifar10.reset();
        valid_cifar10.reset();
    }

    return 0;
};
