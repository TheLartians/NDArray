#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include <lars/index_tuple.h>
using namespace lars;

TEST_CASE( "Index tuple basics", "[IndexTuple]" ) {
  StaticIndexTuple<1,2,3,4> static_index_tuple;
  REQUIRE( static_index_tuple.get<0>() == 1 );
  REQUIRE( static_index_tuple.get<1>() == 2 );
  REQUIRE( static_index_tuple.get<2>() == 3 );
  REQUIRE( static_index_tuple.get<3>() == 4 );
  
  DynamicIndexTuple<4> dynamic_index_tuple = make_dynamic_index_tuple(4,3,2,1);
  REQUIRE( dynamic_index_tuple.get<0>() == 4 );
  REQUIRE( dynamic_index_tuple.get<1>() == 3 );
  REQUIRE( dynamic_index_tuple.get<2>() == 2 );
  REQUIRE( dynamic_index_tuple.get<3>() == 1 );

  REQUIRE_NOTHROW(dynamic_index_tuple.set<2>(42));
  REQUIRE(dynamic_index_tuple.get<2>() == 42);
  
  IndexTuple<StaticIndex<1>,DynamicIndex,StaticIndex<3>,DynamicIndex> mixed_index_tuple;
  REQUIRE_NOTHROW(mixed_index_tuple.set<1>(4));
  REQUIRE_NOTHROW(mixed_index_tuple.set<3>(2));
  REQUIRE( mixed_index_tuple.get<0>() == 1 );
  REQUIRE( mixed_index_tuple.get<1>() == 4 );
  REQUIRE( mixed_index_tuple.get<2>() == 3 );
  REQUIRE( mixed_index_tuple.get<3>() == 2 );

  REQUIRE(dynamic_index_tuple != static_index_tuple);
  REQUIRE(static_index_tuple != dynamic_index_tuple);
  REQUIRE(dynamic_index_tuple != mixed_index_tuple);
  REQUIRE(mixed_index_tuple != static_index_tuple);

  //REQUIRE_THROWS(static_index_tuple = dynamic_index_tuple); only throws in debug mode
  //REQUIRE_THROWS(mixed_index_tuple = dynamic_index_tuple); only throws in debug mode
  REQUIRE_NOTHROW(dynamic_index_tuple = static_index_tuple);
  REQUIRE_NOTHROW(mixed_index_tuple = dynamic_index_tuple);
  REQUIRE_NOTHROW(mixed_index_tuple = static_index_tuple);

  REQUIRE( dynamic_index_tuple.get<0>() == 1 );
  REQUIRE( dynamic_index_tuple.get<1>() == 2 );
  REQUIRE( dynamic_index_tuple.get<2>() == 3 );
  REQUIRE( dynamic_index_tuple.get<3>() == 4 );

  REQUIRE( mixed_index_tuple.get<0>() == 1 );
  REQUIRE( mixed_index_tuple.get<1>() == 2 );
  REQUIRE( mixed_index_tuple.get<2>() == 3 );
  REQUIRE( mixed_index_tuple.get<3>() == 4 );

  REQUIRE(dynamic_index_tuple == static_index_tuple);
  REQUIRE(static_index_tuple == dynamic_index_tuple);
  REQUIRE(mixed_index_tuple == dynamic_index_tuple);

  REQUIRE_NOTHROW(static_index_tuple = dynamic_index_tuple);
  
  REQUIRE( (dynamic_index_tuple + dynamic_index_tuple == StaticIndexTuple<2,4,6,8>()) );
  REQUIRE( (static_index_tuple + static_index_tuple == StaticIndexTuple<2,4,6,8>()) );
  REQUIRE( (dynamic_index_tuple + static_index_tuple == StaticIndexTuple<2,4,6,8>()) );
  REQUIRE( (dynamic_index_tuple * dynamic_index_tuple == StaticIndexTuple<1,4,9,16>()) );
  REQUIRE( (static_index_tuple * static_index_tuple == StaticIndexTuple<1,4,9,16>()) );

  REQUIRE( (decltype(static_index_tuple * static_index_tuple)() == StaticIndexTuple<1,4,9,16>()) );

  static_assert(decltype(static_index_tuple * static_index_tuple)::get<2>() == 9, "compile time evaluation");
  static_assert(decltype(static_index_tuple + static_index_tuple)::get<2>() == 6, "compile time evaluation");
  static_assert(decltype(static_index_tuple * mixed_index_tuple)::get<2>() == 9, "compile time evaluation");
  static_assert(decltype(static_index_tuple + mixed_index_tuple)::get<2>() == 6, "compile time evaluation");

  REQUIRE( (dynamic_index_tuple.slice<0, 2>() == StaticIndexTuple<1,2>()) );
  REQUIRE( (dynamic_index_tuple.slice<2, 4>() != StaticIndexTuple<1,2>()) );
  REQUIRE( (dynamic_index_tuple.slice<2, 4>() == StaticIndexTuple<3,4>()) );

  REQUIRE( (mixed_index_tuple.slice<0, 2>() == StaticIndexTuple<1,2>()) );
  REQUIRE( (mixed_index_tuple.slice<2, 4>() != StaticIndexTuple<1,2>()) );
  REQUIRE( (mixed_index_tuple.slice<2, 4>() == StaticIndexTuple<3,4>()) );
  
  static_assert( decltype( mixed_index_tuple.slice<0, 2>() * static_index_tuple.slice<2, 4>() )::get<0>() == 3, "compile time evaluation");
}

