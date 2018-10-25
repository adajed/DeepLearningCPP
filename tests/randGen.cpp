#include "randGen.h"

UniformGen::UniformGen(unsigned seed, float min, float max)
    : mDist(min, max), mE2(seed)
{
}

UniformGen::UniformGen(unsigned seed) : mDist(-5., 5.), mE2(seed) {}

float UniformGen::operator()() { return mDist(mE2); }
