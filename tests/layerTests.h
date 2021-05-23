#ifndef TESTS_LAYER_TESTS_H_
#define TESTS_LAYER_TESTS_H_

#include "graph.h"
#include "graphdl.h"
#include "pooling.h"
#include "refTensor.h"
#include "utils.h"

#include <functional>
#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>

using namespace graphdl;
using namespace graphdl::core;

using testing::Combine;
using testing::Range;
using testing::ValuesIn;

extern std::vector<MemoryLocation> LOCATIONS;

extern unsigned seed;

/* std::ostream& operator<<(std::ostream& os, layers::PoolingType pooling); */

/* std::ostream& operator<<(std::ostream& os, layers::PaddingType padding); */

/* std::ostream& operator<<(std::ostream& os, layers::DataFormat format); */

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
