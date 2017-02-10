#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include <lars/ndarray.h>
using namespace lars;

#include <iostream>

TEST_CASE( "NDArray Tests", "[NDArray]" ) {
  
   using Array9 = StackNDArray<float,StaticIndexTuple<9>>;
   
   Array9 array;
   array << 4,56,-34,0,5,65,-456,0.1,3.141;
   
   REQUIRE( array.front() == 4 );
   REQUIRE( array.back() == 3.141f );
   REQUIRE( array.min() == -456 );
   REQUIRE( array.max() == 65 );


}


