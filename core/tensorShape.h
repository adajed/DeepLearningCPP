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
    using iterator = std::vector<int>::iterator;

    TensorShape() = default;
    TensorShape(Shape shape);
    TensorShape(std::vector<int> vals);
    TensorShape(const TensorShape& other) = default;
    TensorShape(std::initializer_list<int> list);

    TensorShape& operator=(const TensorShape& other) = default;

    bool operator==(const TensorShape& other) const;
    bool operator!=(const TensorShape& other) const;

    int& operator[](std::size_t pos);
    const int& operator[](std::size_t pos) const;

    unsigned size() const;

    size_t getCount() const;

    operator Shape() const;
    operator std::vector<int>() const;

    iterator begin();
    iterator end();

    //! \fn subshape
    //! \brief Returns shape cut from this shape.
    //! Returned shape is [mDims[start], ..., mDims[start + size - 1]].
    //!
    TensorShape subshape(int start, int size);

  private:
    std::vector<int> mDims;
};

}  // namespace core
}  // namespace graphdl

#endif  // GRAPHDL_CORE_TENSOR_SHAPE_H_
