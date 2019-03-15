/////////////////////////////////////
// toySample
//
// This sample shows the basics of library.
// It uses gradient descent algorithm (GDA) to minimize
// the difference between weights (w) and value 5.
//
/////////////////////////////////////
#include "graphdl.h"
#include "graphdl_ops.h"

#include <iostream>

using namespace graphdl;

// step for GDA
const float STEP = 0.2;

int main()
{
    // computation will be done on CPU
    MemoryLocation loc = MemoryLocation::kHOST;

    // create  weight (a scalar value)
    ITensorPtr w = createWeights("w", {}, constantInitializer(0.), loc);

    // loss for GDA, measures how "far" is weight from 5
    ITensorPtr loss = square(5. - w);

    // library uses automatic differentation to calculate
    // gradients of loss with respect to weights
    ITensorPtr grad = gradients(loss)[w];

    // application of single step of GDA
    ITensorPtr a = assign(w, w - STEP * grad);

    // Before running the graph you need to initialize it.
    // This is reponsible for allocating necessary memory
    // for tensors and initializing their values properly.
    initializeGraph();

    // Each time calculate current loss and apply step of GDA.
    for (int i = 0; i < 10; ++i)
    {
        auto outputs = eval({loss, a}, {});
        std::cout << "step " << i << " : loss " << outputs[0][0];
        HostTensor wHost = w->eval({});
        std::cout << ", weight " << wHost[0] << std::endl;
    }

    return 0;
}
