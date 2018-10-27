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

class LayerTest : public testing::Test
{
   protected:
    virtual void TearDown() override
    {
        dll::core::GraphRegister::getGlobalGraphRegister().clear();
        testing::Test::TearDown();
    }

    using LayerBuilder = std::function<void(const std::vector<HostTensor>&,
                                            const std::vector<HostTensor>&)>;

    bool runTest(const std::vector<RefTensor>& refInputs,
                 const std::vector<RefTensor>& refOutputs,
                 LayerBuilder builder);
};

#endif  // TESTS_LAYER_TESTS_H_
