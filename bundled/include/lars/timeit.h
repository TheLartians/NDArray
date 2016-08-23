
#pragma once

#include <chrono>
#include <iostream>
#include <cmath>
#include <string>
#include <lars/iterators.h>

namespace lars {
  
  template <class Rep,class I> std::string duration_to_string(const std::chrono::duration<Rep,I> &dur){
    using namespace std::chrono;
    
    auto h = duration_cast<hours>(dur);
    auto m = duration_cast<minutes>(dur);
    auto s = duration_cast<seconds>(dur);
    auto ms = duration_cast<milliseconds>(dur);
    auto us = duration_cast<microseconds>(dur);
    auto ns = duration_cast<nanoseconds>(dur);
    
    if(h.count() > 1) return lars::to_string(h.count()) + "h";
    if(m.count() > 1) return lars::to_string(m.count()) + "m";
    if(s.count() > 1) return lars::to_string(s.count()) + "s";
    if(ms.count() > 1) return lars::to_string(ms.count()) + "ms";
    if(us.count() > 1) return lars::to_string(us.count()) + "us";
    return lars::to_string(ns.count()) + "ns";
  }
  
  template<size_t N = 1000, bool calculate_std = false,  class F,typename ... Args> __attribute__ ((noinline)) void timeit(const std::string &name,F f,Args & ... args){
    using I = std::nano;
    
    double squared = 0;
    double total = 0;
    
    std::chrono::steady_clock::time_point start,end;
    
    if(!calculate_std) start = std::chrono::steady_clock::now();
    
    for(auto i UNUSED : range(N)){
      if(calculate_std) start = std::chrono::steady_clock::now();
      f(args ...);
      if(calculate_std){
        end = std::chrono::steady_clock::now();
        auto diff = std::chrono::duration<double, I>( end - start ).count();
        squared += diff * diff;
        total += diff;
      }
    }
    
    if(!calculate_std){
      end = std::chrono::steady_clock::now();
      total = std::chrono::duration<double, I>( end - start ).count();
      std::cout << name << " took " << duration_to_string( std::chrono::duration<double, I>(total/N) ) << std::endl;
    }
    else{
      double std = std::sqrt( squared/N - total/N  * total/N  ) ;
      std::cout << name << " took " << duration_to_string(std::chrono::duration<double, I>(total/N) ) << " +/- " << duration_to_string( std::chrono::duration<double, I>(std)) << " ns" << std::endl;
    }
  }
  
  
}

