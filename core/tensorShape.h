#ifndef DLL_CORE_TENSOR_SHAPE_H_
#define DLL_CORE_TENSOR_SHAPE_H_

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

    size_t count() const;

    operator Shape() const;

    iterator begin();
    iterator end();

   private:
    std::vector<unsigned int> mDims;
};

}  // namespace core
}  // namespace dll

#endif  // DLL_CORE_TENSOR_SHAPE_H_
