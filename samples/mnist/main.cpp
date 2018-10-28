#include "dll.h"
#include "dll_ops.h"
#include "readMNIST.h"

#include <iostream>
#include <random>

const int BATCH_SIZE = 64;
const int NUM_EPOCHS = 1;

using namespace dll;

struct Network
{
    std::map<std::string, ITensorSPtr> inputs;
    std::vector<ITensorSPtr> weights;
    std::vector<ITensorSPtr> modifiers;
    ITensorSPtr output, loss;
};

Network buildNetwork()
{
    ITensorSPtr X = createInput("X", {BATCH_SIZE, 28 * 28});
    ITensorSPtr Y = createInput("Y", {BATCH_SIZE, 10});

    ITensorSPtr W1 = createWeights("W1", {28 * 28, 200});
    ITensorSPtr W2 = createWeights("W2", {200, 100});
    ITensorSPtr W3 = createWeights("W3", {100, 10});

    ITensorSPtr a1 = sigmoid(matmul(X, W1));
    ITensorSPtr a2 = sigmoid(matmul(a1, W2));
    ITensorSPtr a3 = sigmoid(matmul(a2, W3));

    ITensorSPtr loss = reduceSum(square(a3 - Y));

    std::map<ITensorSPtr, ITensorSPtr> grads = gradients(loss);
    std::vector<ITensorSPtr> modifiers;
    for (auto pair : grads)
    {
        ITensorSPtr s = constant(0.1, pair.first->getShape());
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

    HostTensor inputX{nullptr, BATCH_SIZE * 28 * 28};
    HostTensor inputY{nullptr, BATCH_SIZE * 10};
    HostTensor output{nullptr, 0};
    HostTensor lossT{nullptr, 1};
    inputX.values = new float[inputX.count];
    inputY.values = new float[inputY.count];
    lossT.values = new float[lossT.count];

    for (int e = 0; e < NUM_EPOCHS; ++e)
    {
        std::cout << "Epoch " << e << std::endl;
        std::cout << "Number of batches " << mnist.getNumBatches() << std::endl;
        for (int i = 0; i < mnist.getNumBatches(); ++i)
        {
            mnist.getNextBatch(inputX.values, inputY.values);
            eval({net.loss}, {{"X", inputX}, {"Y", inputY}}, {lossT});
            eval(net.modifiers, {{"X", inputX}, {"Y", inputY}},
                 {output, output, output});
            if (i % 100 == 0)
                std::cout << "Loss " << lossT.values[0] << std::endl;
        }
        mnist.reset();
    }

    return 0;
};
