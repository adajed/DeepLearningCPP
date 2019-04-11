#include "layerTests.h"

#ifdef CUDA_AVAILABLE
std::vector<MemoryLocation> LOCATIONS = {
    // clang-format off
    MemoryLocation::kHOST,
    MemoryLocation::kDEVICE
    // clang-format on
};
#else
std::vector<MemoryLocation> LOCATIONS = {
    // clang-format off
    MemoryLocation::kHOST
    // clang-format on
};
#endif

bool compareTensor(const RefTensor& refOutput, const HostTensor& output,
                   float eps)
{
    EXPECT_EQ(refOutput.getCount(), output.size());

    for (size_t i = 0; i < output.size(); ++i)
    {
        Coord c = refOutput.coordAt(i);
        std::string s = "[";
        for (int i = 0; i < int(c.size()) - 1; ++i)
            s += std::to_string(c[i]) + ", ";
        if (c.size() > 0) s += std::to_string(c[c.size() - 1]) + "]";
        EXPECT_NEAR(refOutput.at(i), output[i], eps) << "coord = " << s;
    }

    return true;
}

bool compareTensors(const std::vector<RefTensor>& refOutputs,
                    const std::vector<HostTensor>& outputs, float eps)
{
    EXPECT_EQ(refOutputs.size(), outputs.size());

    bool acc = true;
    for (std::size_t i = 0; i < refOutputs.size(); ++i)
        acc &= compareTensor(refOutputs[i], outputs[i], eps);

    return acc;
}

bool LayerTest::runTest(const std::vector<RefTensor>& refInputs,
                        const std::vector<RefTensor>& refOutputs,
                        LayerBuilder builder, float eps)
{
    // prepare inputs
    std::vector<HostTensor> inputs;
    for (RefTensor ref : refInputs) inputs.push_back(ref.toHostTensor());

    // run graph
    std::vector<HostTensor> outputs = builder(inputs);

    bool ret = compareTensors(refOutputs, outputs, eps);

    return ret;
}
