#include "graphdl.h"
#include "graphdl_ops.h"

#include <iostream>

int main()
{
    graphdl::MemoryLocation loc = graphdl::MemoryLocation::kHOST;
    graphdl::ITensorPtr step = graphdl::scalar(0.2, loc);
    graphdl::ITensorPtr c = graphdl::scalar(5., loc);
    graphdl::ITensorPtr w =
        graphdl::createWeights("w", {}, graphdl::constantInitializer(0.), loc);
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
