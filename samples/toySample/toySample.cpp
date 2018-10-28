#include "dll.h"
#include "dll_ops.h"

#include <iostream>

int main()
{
    dll::ITensorSPtr step = dll::constant(0.2, {1});
    dll::ITensorSPtr c = dll::constant(5., {1});
    dll::ITensorSPtr w = dll::createWeights("w", {1});
    dll::ITensorSPtr loss = dll::square(c - w);
    dll::ITensorSPtr grad = dll::gradients(loss)[w];
    dll::ITensorSPtr a = dll::assign(w, w - step * grad);
    dll::initializeGraph();

    dll::HostTensor lossT{nullptr, 1};
    dll::HostTensor assignT{nullptr, 0};
    lossT.values = new float[1];

    std::vector<dll::ITensorSPtr> calc({loss, a});
    for (int i = 0; i < 10; ++i)
    {
        dll::eval(calc, {}, {lossT, assignT});
        std::cout << "step " << i << " : loss " << lossT.values[0];
        w->eval({}, lossT);
        std::cout << ", weight " << lossT.values[0] << std::endl;
    }

    return 0;
}
