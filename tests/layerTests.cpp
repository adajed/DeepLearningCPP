#include "layerTests.h"

bool compareTensor(const RefTensor& refOutput, const HostTensor& output)
{
    EXPECT_EQ(refOutput.count(), output.count);

    for (std::size_t i = 0; i < output.count; ++i)
        EXPECT_FLOAT_EQ(refOutput.at(i), output.values[i]) << "pos " << i;

    return true;
}

bool compareTensors(const std::vector<RefTensor>& refOutputs, const std::vector<HostTensor>& outputs)
{
    EXPECT_EQ(refOutputs.size(), outputs.size());

    bool acc = true;
    for (std::size_t i = 0; i < refOutputs.size(); ++i)
        acc &= compareTensor(refOutputs[i], outputs[i]);

    return acc;
}

bool LayerTest::runTest(
        const std::vector<RefTensor>& refInputs,
        const std::vector<RefTensor>& refOutputs,
        LayerBuilder builder)
{
    // prepare inputs
    std::vector<HostTensor> inputs;
    for (RefTensor ref : refInputs)
        inputs.push_back(ref.toHostTensor());

    // prepare outputs
    std::vector<HostTensor> outputs;
    for (RefTensor ref : refOutputs)
    {
        HostTensor output{nullptr, ref.count()};
        output.values = new float[output.count];
        outputs.push_back(output);
    }

    // run graph
    builder(inputs, outputs);

    bool ret = compareTensors(refOutputs, outputs);

    for (HostTensor in : inputs)
        delete [] in.values;
    for (HostTensor out : outputs)
        delete [] out.values;

    return ret;
}
