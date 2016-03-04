

#include <chrono>
#include <iostream>

#include "ndarray/ndarray.h"

using namespace lars;

template <template<class,class> class Array,typename S> __attribute__ ((noinline)) Array<int,S> test(S size){
  Array<int,S> array(size);
  
  array.fill(0);
  
  for(auto i:range<unsigned>(3,array.size())) array[i] = i;
  for(auto i:range(array[0].size())) array[0][i] = i;
  
  array[1].fill(2);
  array.transpose()[21].fill(3);
  array[8] = array[0];
  
  array.slice(static_index_tuple<1,20>(), static_index_tuple<5,25>(), static_index_tuple<1,2>()).fill(4);
  array.slice(static_index_tuple<3,2>(), static_index_tuple<5,5>()).fill(5);

  array.transpose().slice(static_index_tuple<1,1>(), static_index_tuple<1,5>()) = array.slice(static_index_tuple<1,1>(), static_index_tuple<1,5>());
  
  return array;
}

template<size_t N = 10000,class F,typename ... Args> __attribute__ ((noinline)) void timeit(const std::string &name,F f,Args & ... args){
  auto start = std::chrono::steady_clock::now();
  
  for(auto i UNUSED : range(N)){
    f(args ...);
  }
  
  auto end = std::chrono::steady_clock::now();
  auto diff = (end - start)/N;
  
  std::cout << name << " took " << std::chrono::duration <double, std::nano> (diff).count() << " ns" << std::endl;
}

int main(){

  {
  static_index_tuple<250,100> size;
  std::cout << "example result:\n" << test<stack_ndarray>(size).slice(static_index_tuple<0,0>(), static_index_tuple<10,100>()) << "\n.\n.\n.\n" << std::endl;
  timeit("stack",[&](){ test<stack_ndarray>(size); });
  }
  
  {
  static_index_tuple<250,100> size;
  timeit("static size heap",[&](){ test<heap_ndarray>(size); });
  }

  {
  dynamic_index_tuple<2> size(250,100);
  timeit("dynamic size heap",[&](){ test<heap_ndarray>(size); });
  }

  return 0;
}
  
