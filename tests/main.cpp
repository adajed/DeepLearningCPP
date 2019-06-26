#include <cstdlib>
#include <ctime>
#include <gtest/gtest.h>
#include <iostream>

unsigned seed = 0;

unsigned getSeed(int argc, char** argv)
{
    int len = strlen("--seed=");
    for (int i = 0; i < argc; ++i)
        if (strncmp("--seed=", argv[i], len) == 0)
            return unsigned(atoi(argv[i] + len));

    return unsigned(rand());
}

bool getFilter(int argc, char** argv, std::string& filter)
{
    int len = strlen("--filter=");
    for (int i = 0; i < argc; ++i)
    {
        if (strncmp("--filter=", argv[i], len) == 0)
        {
            filter = std::string(argv[i] + len);
            return true;
        }
    }

    return false;
}

bool getRange(int argc, char** argv, int& start, int& end)
{
    int len = strlen("--range=");
    for (int i = 0; i < argc; ++i)
    {
        if (strncmp("--range=", argv[i], len) == 0)
        {
            std::string str = std::string(argv[i] + len);
            int pos = str.find("-");
            if (pos != std::string::npos)
            {
                start = std::stoi(str.substr(0, pos));
                end = std::stoi(str.substr(pos + 1, std::string::npos));
            }
            else
                start = end = std::stoi(str);
            return true;
        }
    }

    return false;
}

int main(int argc, char** argv)
{
    srand(time(NULL));
    seed = getSeed(argc, argv);

    std::string layer;
    int start, end;

    bool bf = getFilter(argc, argv, layer);
    bool br = getRange(argc, argv, start, end);

    if (bf)
    {
        std::string filter = "";
        if (br)
            for (int i = start; i <= end; ++i)
                filter += "LayerTest/" + layer + "Test.testAPI/" + std::to_string(i) + ":";
        else
            filter = "LayerTest/" + layer + "Test.testAPI/*";

        ::testing::GTEST_FLAG(filter) = filter;
    }

    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();

    std::cout << "SEED = " << seed << std::endl;

    return ret;
}
