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
    ITensorPtr a = conv2D(x, k, {1, 1}, "SAME", "NHWC");
    return relu(maxPool2D(a, {2, 2}, {2, 2}, "VALID", "NHWC"));
}

ComputationalGraph buildNetwork()
{
    MemoryLocation loc = MemoryLocation::kDEVICE_IF_ENABLED;

    // inputs
    ITensorPtr X = createInput("X", {BATCH_SIZE, 32, 32, 3}, loc);
    ITensorPtr Y = createInput("Y", {BATCH_SIZE, 10}, loc);

    ITensorPtr a = X;

    a = create_conv2D(a, 8, {3, 3}, {1, 1}, "SAME", "NHWC", "conv1");
    a = maxPool2D(a, {2, 2}, {2, 2}, "VALID", "NHWC");
    a = relu(a);

    a = create_conv2D(a, 16, {3, 3}, {1, 1}, "SAME", "NHWC", "conv2");
    a = maxPool2D(a, {2, 2}, {2, 2}, "VALID", "NHWC");
    a = relu(a);

    a = create_conv2D(a, 32, {3, 3}, {1, 1}, "SAME", "NHWC", "conv3");
    a = maxPool2D(a, {2, 2}, {2, 2}, "VALID", "NHWC");
    a = relu(a);

    a = create_conv2D(a, 64, {3, 3}, {1, 1}, "SAME", "NHWC", "conv4");

    a = reshape(a, {BATCH_SIZE, 64 * 4 * 4});
    a = create_matmulAndAddBias(a, 128, "dense1");
    a = relu(a);
    ITensorPtr logits = create_matmulAndAddBias(a, 10, "dense2");
    ITensorPtr prob = softmax_c(logits, 1);

    ITensorPtr loss = reduceMean(softmax_cross_entropy_with_logits(logits, Y));

    ITensorPtr opt =
        train::adam(LEARNING_RATE, 0.9, 0.999, 10e-8)->optimize(loss);

    ComputationalGraph net;
    net.inputs = {{"X", X}, {"Y", Y}};
    net.weights = {};
    net.output = prob;
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
