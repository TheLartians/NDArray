#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include <lars/matrix.h>
using namespace lars;

template <size_t size> void SquareMatrixTests(){
  using Mat = Matrix<float, size, size> ;
  
  Matrix<float, size, size> matrix;
  matrix.for_all_indices([&](DynamicIndexTuple<2> idx){
    matrix(idx) = idx.get<0>() > idx.get<1>() ? (idx.get<0>() + 1.)/(idx.get<1>() + 1) * idx.get<1>() :
                  (idx.get<1>() + 1.)*(idx.get<0>() + 1) + idx.get<0>();
  });
  
  REQUIRE((matrix.inverse() * matrix > 1e-4) == Mat::create_identity());

}

TEST_CASE( "Matrix Tests", "[Matrix]" ) {
  SquareMatrixTests<1>();
  SquareMatrixTests<2>();
  SquareMatrixTests<3>();
  SquareMatrixTests<4>();
  SquareMatrixTests<8>();
  SquareMatrixTests<21>();
  SquareMatrixTests<56>();
  SquareMatrixTests<100>();
}
