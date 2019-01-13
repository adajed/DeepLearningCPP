#ifndef TESTS_REF_TENSOR_H_
#define TESTS_REF_TENSOR_H_

#include "graphdl.h"
#include "randGen.h"
#include "tensorShape.h"

#include <initializer_list>
#include <ostream>
#include <vector>

using namespace graphdl;
using namespace graphdl::core;

class Coord
{
  public:
    Coord(const std::vector<unsigned>& values);
    Coord(std::initializer_list<unsigned> list);

    unsigned size() const;

    unsigned& operator[](size_t pos);
    const unsigned& operator[](size_t pos) const;

  private:
    std::vector<unsigned> mValues;
};

class Coord_iterator
{
  public:
    Coord_iterator(Coord c, Coord shape);
    Coord_iterator(const Coord_iterator& it) = default;
    Coord_iterator& operator=(const Coord_iterator& it) = default;

    Coord_iterator operator++();
    Coord_iterator operator++(int junk);

    bool operator==(const Coord_iterator& it) const;
    bool operator!=(const Coord_iterator& it) const;

    Coord& operator()();

  private:
    Coord mCoord;
    Coord mShape;
};

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
    float& operator[](const Coord& c);
    const float& operator[](const Coord& c) const;

    Coord_iterator begin();
    Coord_iterator end();

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
