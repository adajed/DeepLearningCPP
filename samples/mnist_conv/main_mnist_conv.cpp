#include "graphdl.h"
#include "graphdl_ops.h"
#include "graphdl_train.h"
#include "readMNIST.h"
#include "utils.h"

#include <iostream>
#include <random>

const int BATCH_SIZE = 64;  // how many samples per computation
const int NUM_EPOCHS = 1;  // # of runs over whole dataset
const int PRINT_EVERY = 100;  // after how many batches print info
const float LEARNING_RATE = 0.001;  // learning parameter to the optimizer

#define Q(x) std::string(#x)
#define QUOTE(x) Q(x)

const std::string TRAIN_IMAGES_PATH =
    QUOTE(PROJECT_DIR) + "/samples/mnist_conv/train-images-idx3-ubyte";
const std::string TRAIN_LABELS_PATH =
    QUOTE(PROJECT_DIR) + "/samples/mnist_conv/train-labels-idx1-ubyte";
const std::string VALID_IMAGES_PATH =
    QUOTE(PROJECT_DIR) + "/samples/mnist_conv/t10k-images-idx3-ubyte";
const std::string VALID_LABELS_PATH =
    QUOTE(PROJECT_DIR) + "/samples/mnist_conv/t10k-labels-idx1-ubyte";

#undef Q
#undef QUOTE

using namespace graphdl;

ComputationalGraph buildNetwork()
{
    MemoryLocation loc = MemoryLocation::kDEVICE_IF_ENABLED;

    ITensorPtr X = createInput("X", {BATCH_SIZE, 28, 28, 1}, loc);
    ITensorPtr Y = createInput("Y", {BATCH_SIZE, 10}, loc);

    ITensorPtr a;
    a = create_conv2D(X, 4, {3, 3}, {1, 1}, "SAME", "NHWC", "conv1");
    a = maxPool2D(a, {2, 2}, {2, 2}, "SAME", "NHWC");
    a = relu(a);

    a = create_conv2D(a, 16, {3, 3}, {1, 1}, "SAME", "NHWC", "conv2");
    a = maxPool2D(a, {2, 2}, {2, 2}, "SAME", "NHWC");
    a = relu(a);

    a = create_conv2D(a, 32, {3, 3}, {1, 1}, "SAME", "NHWC", "conv3");
    a = reshape(a, {BATCH_SIZE, 32 * 7 * 7});
    a = relu(create_matmulAndAddBias(a, 128, "dense1"));
    a = create_matmulAndAddBias(a, 10, "dense2");
    a = softmax_c(a, 1);

    ITensorPtr loss = neg(reduceSum(Y * log(a))) / float(BATCH_SIZE);

    ITensorPtr opt =
        train::adam(LEARNING_RATE, 0.9, 0.999, 10e-8)->optimize(loss);

    ComputationalGraph net;
    net.inputs = {{"X", X}, {"Y", Y}};
    net.weights = {};
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
        std::cout << "Number of batches " << train_mnist.getNumBatches()
                  << std::endl;
        for (int i = 0; i < train_mnist.getNumBatches(); ++i)
        {
            auto batch = train_mnist.getNextBatch();
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
        for (int i = 0; i < valid_mnist.getNumBatches(); ++i)
        {
            auto batch = valid_mnist.getNextBatch();
            auto outputs = eval({net.loss, net.output},
                                {{"X", batch[0]}, {"Y", batch[1]}});
            losses.push_back(outputs[0][0]);
            accs.push_back(calcNumCorrect(batch[1], outputs[1], BATCH_SIZE));
        }
        std::cout << "Valid. loss " << mean(losses) << ", Valid. acc "
                  << accuracy(accs, BATCH_SIZE) << std::endl;
        train_mnist.reset();
        valid_mnist.reset();
    }

    return 0;
};
