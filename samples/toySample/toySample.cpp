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

    for (int i = 0; i < 10; ++i)
    {
        auto outputs = graphdl::eval({loss, a}, {});
        std::cout << "step " << i << " : loss " << outputs[0][0];
        graphdl::HostTensor wHost = w->eval({});
        std::cout << ", weight " << wHost[0] << std::endl;
    }

    return 0;
}
