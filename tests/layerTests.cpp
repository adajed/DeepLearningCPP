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

std::ostream& operator<<(std::ostream& os, MemoryLocation loc)
{
    if (loc == MemoryLocation::kHOST) return os << "HOST";
    return os << "DEVICE";
}

std::ostream& operator<<(std::ostream& os, layers::PoolingType pooling)
{
    if (pooling == layers::PoolingType::kMAX) return os << "MAX";
    return os << "AVERAGE";
}

std::ostream& operator<<(std::ostream& os, layers::PaddingType padding)
{
    if (padding == layers::PaddingType::kSAME) return os << "SAME";
    return os << "VALID";
}

std::ostream& operator<<(std::ostream& os, layers::DataFormat format)
{
    if (format == layers::DataFormat::kNHWC) return os << "NHWC";
    return os << "NCHW";
}

bool compareTensor(const RefTensor& refOutput, const HostTensor& output,
                   float eps, int tensorNum)
{
    EXPECT_EQ(refOutput.getCount(), output.size())
        << refOutput.getCount() << " and " << output.size();

    for (size_t i = 0; i < output.size(); ++i)
    {
        Coord c = refOutput.coordAt(i);
        std::string s = "[";
        for (int i = 0; i < int(c.size()) - 1; ++i)
            s += std::to_string(c[i]) + ", ";
        if (c.size() > 0) s += std::to_string(c[c.size() - 1]) + "]";
        EXPECT_NEAR(refOutput.at(i), output[i], eps)
            << "tensor = " << tensorNum << ", coord = " << s;
    }

    return true;
}

bool compareTensors(const std::vector<RefTensor>& refOutputs,
                    const std::vector<HostTensor>& outputs, float eps)
{
    EXPECT_EQ(refOutputs.size(), outputs.size());

    bool acc = true;
    for (std::size_t i = 0; i < refOutputs.size(); ++i)
        acc &= compareTensor(refOutputs[i], outputs[i], eps, i);

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
