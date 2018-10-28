#include "dll.h"
#include "dll_ops.h"

#include <iostream>
#include <random>

const int BATCH_SIZE = 64;

using namespace dll;

int main()
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
    std::vector<ITensorSPtr> calc;
    for (auto pair : grads)
    {
        ITensorSPtr s = constant(0.1, pair.first->getShape());
        calc.push_back(assign(pair.first, pair.first - s * pair.second));
    }
    calc.push_back(loss);

    initializeGraph();

    HostTensor inputX{nullptr, BATCH_SIZE * 28 * 28};
    HostTensor inputY{nullptr, BATCH_SIZE * 10};
    HostTensor output{nullptr, 0};
    HostTensor lossT{nullptr, 1};
    inputX.values = new float[inputX.count];
    inputY.values = new float[inputY.count];
    lossT.values = new float[lossT.count];

    std::vector<HostTensor> outs({output, output, output, lossT});

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(-1., 1.);
    for (std::size_t pos = 0; pos < inputX.count; ++pos)
        inputX.values[pos] = dist(e2);
    for (std::size_t pos = 0; pos < inputY.count; ++pos)
        inputY.values[pos] = dist(e2);

    for (int i = 0; i < 100; ++i)
    {
        eval(calc, {{"X", inputX}, {"Y", inputY}}, outs);
        std::cout << "step " << i << " : loss " << outs[3].values[0]
                  << std::endl;
    }

    return 0;
};
