#ifndef TESTS_RAND_GEN_H_
#define TESTS_RAND_GEN_H_

#include <random>

class RandGen
{
   public:
    virtual float operator()() = 0;
};

class UniformGen : public RandGen
{
   public:
    UniformGen(unsigned int seed, float min, float max);
    UniformGen(unsigned int seed);

    virtual float operator()() override;

   private:
    std::uniform_real_distribution<> mDist;
    std::mt19937 mE2;
};

#endif  // TESTS_RAND_GEN_H_
