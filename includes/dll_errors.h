#ifndef DLL_ERRORS_H_
#define DLL_ERRORS_H_

#include <exception>

namespace dll
{
namespace errors
{

class MemoryAllocationError : public std::exception
{
    virtual const char* what() const throw()
    {
        return "MemoryAllocationError";
    }
};

class WeightsInitializationError : public std::exception
{
    virtual const char* what() const throw()
    {
        return "WeightsInitializationError";
    }
};

} // namespace errors
} // namespace dll

#endif // DLL_ERRORS_H_
