#ifndef DLL_CORE_TENSOR_SHAPE_H_
#define DLL_CORE_TENSOR_SHAPE_H_

#include <initializer_list>

#include "dll.h"

namespace dll
{
namespace core
{

class TensorShape
{
public:
    using iterator = std::vector<unsigned int>::iterator;

    TensorShape(const Shape& shape);
    TensorShape(const TensorShape& other);
    TensorShape(std::initializer_list<unsigned> list);

    bool operator ==(const TensorShape& other) const;

    unsigned& operator [](std::size_t pos);
    const unsigned& operator [] (std::size_t pos) const;

    unsigned size() const;

    size_t count() const;

    operator Shape() const;

    iterator begin();
    iterator end();

private:
    std::vector<unsigned int> mDims;
};

} // namespace core
} // namespace dll

#endif // DLL_CORE_TENSOR_SHAPE_H_
