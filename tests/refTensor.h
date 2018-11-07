#ifndef TESTS_REF_TENSOR_H_
#define TESTS_REF_TENSOR_H_

#include "graphdl.h"
#include "randGen.h"
#include "tensorShape.h"

#include <ostream>

using namespace graphdl;
using namespace graphdl::core;

class RefTensor
{
  public:
    RefTensor();
    RefTensor(const TensorShape& shape);

    //! \fn at
    //! \brief Returns value given its linear coordinate.
    //!
    float& at(std::size_t pos);
    const float& at(std::size_t pos) const;

    //! \fn operator []
    //! \brief Return value given its multidimensional coordinate.
    //!
    float& operator[](const std::vector<unsigned int>& point);
    const float& operator[](const std::vector<unsigned int>& point) const;

    //! \fn getCount
    std::size_t getCount() const;

    //! \fn shape
    TensorShape shape() const;

    //! \fn fillRandomly
    void fillRandomly(RandGen& gen);

    //! \fn toHostTensor
    HostTensor toHostTensor();

  private:
    std::vector<float> mValues;
    std::size_t mCount;
    TensorShape mShape;
};

std::ostream& operator<<(std::ostream&, const RefTensor&);

#endif  // TESTS_REF_TENSOR_H_
