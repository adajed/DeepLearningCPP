#ifndef TESTS_LAYER_TESTS_H_
#define TESTS_LAYER_TESTS_H_

#include <gtest/gtest.h>
#include <functional>

#include "graph.h"
#include "graphdl.h"
#include "refTensor.h"

using namespace graphdl;
using namespace graphdl::core;

using testing::Combine;
using testing::Range;
using testing::ValuesIn;

using Vec = std::vector<unsigned>;
using HostVec = std::vector<HostTensor>;

class LayerTest : public testing::Test
{
   protected:
    virtual void TearDown() override
    {
        getGraphRegister().clear();
        testing::Test::TearDown();
    }

    using LayerBuilder = std::function<HostVec(const HostVec&)>;

    bool runTest(const std::vector<RefTensor>& refInputs,
                 const std::vector<RefTensor>& refOutputs, LayerBuilder builder,
                 float eps = 10e-6);
};

#endif  // TESTS_LAYER_TESTS_H_
