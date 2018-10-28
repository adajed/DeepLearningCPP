#ifndef TESTS_LAYER_TESTS_H_
#define TESTS_LAYER_TESTS_H_

#include <gtest/gtest.h>
#include <functional>

#include "dll.h"
#include "graph.h"
#include "refTensor.h"

using namespace dll;
using namespace dll::core;

using testing::Combine;
using testing::ValuesIn;

using Vec = std::vector<unsigned>;
using HostVec = std::vector<HostTensor>;

class LayerTest : public testing::Test
{
   protected:
    virtual void TearDown() override
    {
        dll::core::GraphRegister::getGlobalGraphRegister().clear();
        testing::Test::TearDown();
    }

    using LayerBuilder = std::function<void(const HostVec&, const HostVec&)>;

    bool runTest(const std::vector<RefTensor>& refInputs,
                 const std::vector<RefTensor>& refOutputs, LayerBuilder builder,
                 float eps = 10e-6);
};

#endif  // TESTS_LAYER_TESTS_H_
