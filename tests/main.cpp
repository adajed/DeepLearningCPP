#include <cstdlib>
#include <ctime>
#include <gtest/gtest.h>
#include <iostream>

unsigned seed = 0;

unsigned getSeed(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i)
        if (strncmp("--seed=", argv[i], 7) == 0)
            return unsigned(atoi(argv[i] + 7));

    return unsigned(rand());
}

int main(int argc, char** argv)
{
    srand(time(NULL));
    seed = getSeed(argc, argv);

    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();

    std::cout << "SEED = " << seed << std::endl;

    return ret;
}
