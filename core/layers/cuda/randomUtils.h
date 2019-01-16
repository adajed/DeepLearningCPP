namespace graphdl
{
namespace core
{
namespace cuda
{

void uniformRandom(float* memory, size_t size, float min, float max, size_t seed);

void normalRandom(float* memory, size_t size, float mean, float stddev, size_t seed);

}
}
}
