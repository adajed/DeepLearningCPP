#include <gtest/gtest.h>
#include "dll.h"

TEST(InitializeTest, good)
{
    dll::initializeGraph();
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
