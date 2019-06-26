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
    Coord(const std::vector<int>& values);
    Coord(std::initializer_list<int> list);

    Coord operator+(const Coord& c) const;

    unsigned size() const;

    int& operator[](size_t pos);
    const int& operator[](size_t pos) const;

  private:
    std::vector<int> mValues;
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

    int& operator[](size_t pos);
    const int& operator[](size_t pos) const;

  private:
    Coord mCoord;
    Coord mShape;
};

//! \brief Tests whether coordinate is inside shape
bool isInside(const Coord& c, const TensorShape& shape);

class RefTensor
{
  public:
    RefTensor();
    RefTensor(const TensorShape& shape);
    RefTensor(const TensorShape& shape, RandGen& gen);

    //! \fn at
    //! \brief Returns value given its linear coordinate.
    //!
    float& at(size_t pos);
    const float& at(size_t pos) const;

    //! \fn operator []
    //! \brief Return value given its multidimensional coordinate.
    //!
    float& operator[](const Coord& c);
    const float& operator[](const Coord& c) const;

    Coord coordAt(size_t pos) const;

    Coord_iterator begin();
    Coord_iterator end();

    RefTensor slice(Coord start, const TensorShape& shape) const;

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

Coord_iterator shapeBegin(const TensorShape& shape);

Coord_iterator shapeEnd(const TensorShape& shape);

std::ostream& operator<<(std::ostream&, const RefTensor& t);

std::ostream& operator<<(std::ostream&, const Coord& c);

#endif  // TESTS_REF_TENSOR_H_
