
#pragma once

#include <iterator>

namespace lars {
  
#if defined(__GNUC__)
#  define UNUSED __attribute__ ((unused))
#elif defined(_MSC_VER)
#  define UNUSED __pragma(warning(suppress:4100))
#else
#  define UNUSED
#endif
  
  
  template<class T> class range_wrapper {
  
    const T r_start,r_end,r_increment;

  public:
    
    class const_iterator : public std::iterator<std::input_iterator_tag, T>{
    private:
      T current;
      const T increment;
      
    public:
      const_iterator(const T &current,const T &increment) : current(current),increment(increment){ }
      
      T operator*()const{ return current; }
      
      const_iterator &operator++() { current+=increment; return *this; }
      const_iterator &operator++(int) { const_iterator range_copy(*this); current+=increment; return range_copy; }
      bool operator!=(const const_iterator &rhs) { return (increment>=0 && (current < rhs.current)) || (increment<0 && (current > rhs.current)); }
    };
    
    
    range_wrapper(const T &r_start, const T &r_end,const T &r_increment) : r_start(r_start), r_end(r_end), r_increment(r_increment) {

    }

    using const_reverse_iterator = const_iterator;

    const_iterator begin() const { return const_iterator(r_start,r_increment); }
    const_iterator end()   const { return const_iterator(r_end  ,r_increment); }

    const_reverse_iterator rbegin() const { return const_reverse_iterator(r_end-1    ,-r_increment); }
    const_reverse_iterator rend()   const { return const_reverse_iterator(r_start-1  ,-r_increment); }
    
    range_wrapper<T> operator+(const T &d){ return range_wrapper<T>(r_start + d,r_end + d,r_increment); }
    range_wrapper<T> operator*(const T &m){ return range_wrapper<T>(r_start * m,r_end * m,r_increment * m); }
    
  };

  template<class T> range_wrapper<T> range(const T &start, const T &end, const T & increment) { return range_wrapper<T>(start, end, increment); }
  template<class T> range_wrapper<T> range(const T &start, const T &end) { return range_wrapper<T>(start, end, 1); }
  template<class T> range_wrapper<T> range(const T &end) { return range_wrapper<T>(0, end, 1); }

}
 
