#include "graphdl.h"
#include "graphdl_ops.h"

#include <iostream>

int main()
{
    graphdl::ITensorPtr step = graphdl::constant(0.2, {1});
    graphdl::ITensorPtr c = graphdl::constant(5., {1});
    graphdl::ITensorPtr w = graphdl::createWeights("w", {1});
    graphdl::ITensorPtr loss = graphdl::square(c - w);
    graphdl::ITensorPtr grad = graphdl::gradients(loss)[w];
    graphdl::ITensorPtr a = graphdl::assign(w, w - step * grad);
    graphdl::initializeGraph();

    graphdl::HostTensor lossT(1);
    graphdl::HostTensor assignT(0);
    lossT[0] = 1;

    std::vector<graphdl::ITensorPtr> calc({loss, a});
    for (int i = 0; i < 10; ++i)
    {
        graphdl::eval(calc, {}, {lossT, assignT});
        std::cout << "step " << i << " : loss " << lossT[0];
        w->eval({}, lossT);
        std::cout << ", weight " << lossT[0] << std::endl;
    }

    return 0;
}
