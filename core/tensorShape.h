#ifndef GRAPHDL_CORE_TENSOR_SHAPE_H_
#define GRAPHDL_CORE_TENSOR_SHAPE_H_

#include "graphdl.h"

#include <initializer_list>

namespace graphdl
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

    TensorShape& operator=(const TensorShape& other);

    bool operator==(const TensorShape& other) const;
    bool operator!=(const TensorShape& other) const;

    unsigned& operator[](std::size_t pos);
    const unsigned& operator[](std::size_t pos) const;

    unsigned size() const;

    size_t getCount() const;

    operator Shape() const;

    iterator begin();
    iterator end();

  private:
    std::vector<unsigned int> mDims;
};

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_TENSOR_SHAPE_H_
