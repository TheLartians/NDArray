

#include <chrono>
#include <iostream>

#include "ndarray/ndarray.h"

using namespace lars;

template <template<class,class> class Array,typename S> __attribute__ ((noinline)) Array<int,S> test(S size){
  Array<int,S> array(size);
  
  array.fill(0);
  array[0].fill(1);
  array[1].fill(2);
  array.transpose()[21].fill(3);
  array[8] = array[0];
  
  array.slice(fixed_index_tuple<1,20>(), fixed_index_tuple<5,25>(), fixed_index_tuple<1,2>()) = 4;
  array.slice(fixed_index_tuple<3,2>(), fixed_index_tuple<5,5>()) = 5;
  
  return array;
}

template<size_t N = 1000000,class F,typename ... Args> void timeit(const std::string &name,F f,Args & ... args){
  auto start = std::chrono::steady_clock::now();
  
  for(auto i UNUSED : range(N)){
    f(args ...);
  }
  
  auto end = std::chrono::steady_clock::now();
  auto diff = (end - start)/N;
  
  std::cout << name << " took " << std::chrono::duration <double, std::nano> (diff).count() << " ns" << std::endl;
}


int main(){
  
  fixed_index_tuple<10,100> size;
  std::cout << "Example result:\n" << test<stack_ndarray>(size) << "\n.\n.\n.\n" << std::endl;

  //*
  {
  fixed_index_tuple<200,100> size;
  timeit("stack",[&](){ test<stack_ndarray>(size); });
  }
  //*/
  
  {
  fixed_index_tuple<200,100> size;
  timeit("fixed size heap",[&](){ test<heap_ndarray>(size); });
  }

  {
  dynamic_index_tuple<2> size(200,100);
  timeit("dynamic size heap",[&](){ test<heap_ndarray>(size); });
  }

  return 0;
}
  
