

#include <chrono>
#include <iostream>

#include "ndarray/ndarray.h"

using namespace lars;


int main(){

  // Create a resizable 2D array of doubles
  ndarray<double, dynamic_index_tuple<2>> array;
  
  // Resize the array
  array.resize(10,20);
  
  // Fill it with zeros
  array.fill(0);
  
  // Set the second item in the first row to one
  array[0][1] = 1;
  
  // Fill the third row with twos
  array[2].fill(2);
  
  // Fill the third colums with threes
  array.transpose()[2].fill(3);
  
  // Fill a slice at the position (3,3) with sizes (3,4) with fours
  array.slice(static_index_tuple<3,3>() , static_index_tuple<3,4>()).fill(4);
  
  // Fill a slice at the position (5,8) with sizes (3,4) and step size (2,3) with fives
  array.slice(dynamic_index_tuple<2>(5,8) , dynamic_index_tuple<2>(3,4), dynamic_index_tuple<2>(2,3)).fill(5);
  
  // Show the array
  std::cout << array << std::endl;
  
  return 0;
}
  
