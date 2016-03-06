
#pragma once

#include <tuple>
#include <array>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <ostream>

namespace lars {
  
struct dynamic_index{
  size_t value = 0;
  constexpr dynamic_index(){}
  constexpr dynamic_index(size_t _value):value(_value){}
  operator size_t()const{ return value; }
  constexpr static bool is_dynamic = true;
};

template <size_t _value> struct static_index{
  constexpr static_index(){}
  static const size_t value = _value;
  operator size_t()const{ return _value; }
  constexpr static bool is_dynamic = false;
};

template <typename ... Indices> class index_tuple;

template <typename Lhs,typename Rhs,typename F> struct reducer:public F{
  const Lhs & lhs;
  const Rhs & rhs;
  
  reducer(const Lhs & _lhs,const Rhs & _rhs):lhs(_lhs),rhs(_rhs){ static_assert(Lhs::size() == Rhs::size(), "index tuple size doesn't match"); }
  
  template <size_t N,typename Enable = void> struct make_result_type;
  
  template <size_t N> struct make_result_type<N,typename std::enable_if<N!=0 && (Lhs::template is_dynamic<N-1>() || Rhs::template is_dynamic<N-1>())>::type>{
    using type = typename make_result_type<N-1>::type::push_back_dynamic_type;
  };
  
  template <size_t N> struct make_result_type<N,typename std::enable_if<N!=0 && !(Lhs::template is_dynamic<N-1>() || Rhs::template is_dynamic<N-1>())>::type>{
    using type = typename make_result_type<N-1>::type::template push_back_static_type<F::reduce(Lhs::template get<N-1>(),Rhs::template get<N-1>())>;
  };
  
  template <size_t N> struct make_result_type<N,typename std::enable_if<N==0>::type>{ using type = index_tuple<>; };
  
  using result_type = typename make_result_type<Lhs::size()>::type;
  
  template <size_t Idx> void operator()(dynamic_index & value){
    value = F::reduce(lhs.template get<Idx>(),rhs.template get<Idx>());
  }
  
  template <size_t Idx,size_t value> void operator()(static_index<value> &){  }
  
};


template <typename ... Indices> class index_tuple:public std::tuple<Indices...>{
  
public:
  
  constexpr static size_t size(){ return sizeof...(Indices); }
  
  template <size_t Idx = 0,typename F = void> typename std::enable_if<Idx+1 < size()>::type apply_template(F & f)const{
    f.template operator()<Idx>(std::get<Idx>(*this));
    apply_template<Idx+1>(f);
  }
  
  template <size_t Idx = 0,typename F = void> typename std::enable_if<Idx+1 == size()>::type apply_template(F & f)const{
    f.template operator()<Idx>(std::get<Idx>(*this));
  }
  
  template <size_t Idx = 0,typename F = void> typename std::enable_if<Idx+1 < size()>::type apply(const F & f)const{
    f(Idx,std::get<Idx>(*this).value);
    apply<Idx+1>(f);
  }
  
  template <size_t Idx = 0,typename F = void> typename std::enable_if<Idx+1 == size()>::type apply(const F & f)const{
    f(Idx,std::get<Idx>(*this).value);
  }
  
  template <size_t Idx = 0,typename F = void> typename std::enable_if<Idx+1 < size()>::type apply_template(F & f){
    f.template operator()<Idx>(std::get<Idx>(*this));
    apply_template<Idx+1>(f);
  }
  
  template <size_t Idx = 0,typename F = void> typename std::enable_if<Idx+1 == size()>::type apply_template(F & f){
    f.template operator()<Idx>(std::get<Idx>(*this));
  }
  
  template <size_t Idx = 0,typename F = void> typename std::enable_if<Idx+1 < size()>::type apply(const F & f){
    f(Idx,std::get<Idx>(*this));
    apply<Idx+1>(f);
  }
  
  template <size_t Idx = 0,typename F = void> typename std::enable_if<Idx+1 == size()>::type apply(const F & f){
    f(Idx,std::get<Idx>(*this));
  }

  template <size_t Idx = 0,typename F = void> typename std::enable_if<size()+0*Idx == 0>::type apply(const F & f)const{ }
  template <size_t Idx = 0,typename F = void> typename std::enable_if<size()+0*Idx == 0>::type apply_template(F & f)const{ }
  
  template <size_t Idx> using element_type = typename std::tuple_element<Idx, std::tuple<Indices...> >::type;
  
  template <size_t Idx> constexpr static typename std::enable_if<(Idx<size()),bool>::type is_dynamic(){ return std::tuple_element<Idx, std::tuple<Indices...> >::type::is_dynamic; }
  
private:
  
  struct mul_reduce{ constexpr static size_t reduce(size_t a,size_t b){ return a*b; } };
  struct sum_reduce{ constexpr static size_t reduce(size_t a,size_t b){ return a+b; } };
  struct div_reduce{ constexpr static size_t reduce(size_t a,size_t b){ return a/b; } };
  struct dif_reduce{ constexpr static size_t reduce(size_t a,size_t b){ return a-b; } };
  
  template <typename Args,size_t Begin = 0> struct setter{
    const Args & values;
    setter(const Args &_values):values(_values){}
    
    template <size_t Idx> using enable_if_dynamic = typename std::enable_if<std::tuple_element<Idx, Args>::type::is_dynamic>::type;
    template <size_t Idx> using enable_if_not_dynamic = typename std::enable_if<!std::tuple_element<Idx, Args>::type::is_dynamic>::type;
    
    template <size_t Idx>  void operator()(dynamic_index & value)const{ value = size_t(std::get<Idx + Begin>(values)); }
    
    template <size_t Idx,size_t value> enable_if_dynamic<Idx> operator()(static_index<value> &v)const{
      if(std::get<Idx + Begin>(values) != value) throw std::runtime_error("changing static index");
    }
    
    template <size_t Idx,size_t value> enable_if_not_dynamic<Idx> operator()(static_index<value> &v)const{
      static_assert( std::tuple_element<Idx + Begin, Args>::type::value == value, "changing static index");
    }

  };
  
  template <size_t Idx,class Ret> using enable_if_dynamic = typename std::enable_if<std::tuple_element<Idx, std::tuple<Indices...>>::type::is_dynamic,Ret>::type;
  template <size_t Idx,class Ret> using enable_if_not_dynamic = typename std::enable_if<(!std::tuple_element<Idx, std::tuple<Indices...>>::type::is_dynamic),Ret>::type;
  
public:
  
  index_tuple(){}
  
  template <typename ... Args> index_tuple(const index_tuple<Args...> &other){ set(other); }
  template <typename ... Args> index_tuple(Args ... args){ set(args...); }
  
  template <typename ... Args> void set(const std::tuple<Args...> &args){
    static_assert(sizeof...(Args) == size(), "index tuple size doesn't match size");
    setter<std::tuple<Args...>> value_setter(args);
    apply_template(value_setter);
  }
  
  template <typename Arg> index_tuple & operator=(const Arg &other){
    set(other);
    return *this;
  }
  
  template <typename F,typename Rhs> typename reducer<index_tuple<Indices...>,Rhs,F>::result_type reduce(Rhs rhs)const{
    typename reducer<index_tuple<Indices...>,Rhs,F>::result_type result;
    reducer<index_tuple<Indices...>,Rhs,F> reducer(*this,rhs);
    result.apply_template(reducer);
    return result;
  }
  
  template <typename ... Args> void set(size_t first, Args ... args){ set(std::make_tuple( dynamic_index(first), dynamic_index(args) ...) ); }
  
  template <size_t Idx> void set(size_t value){ std::get<Idx>(*this) = value; }
  template <size_t ... Idx> void set(){ set(std::make_tuple( static_index<Idx>() ... ) ); }
  
  template <size_t Idx> enable_if_dynamic<Idx, size_t> get()const{ return std::get<Idx>(*this).value; }
  template <size_t Idx> enable_if_not_dynamic<Idx, size_t> static constexpr get(){ return std::tuple_element<Idx, std::tuple<Indices...> >::type::value; }
  
  template <typename> struct make_append;
  template <typename... OtherIndices> struct make_append<index_tuple<OtherIndices...>> { using type = index_tuple<Indices..., OtherIndices...>; };
  template <typename Other> using append_type = typename make_append<Other>::type;
  
  template <typename... OtherIndices> index_tuple<Indices...,OtherIndices...> append(const index_tuple<OtherIndices...> &other)const{
    index_tuple<Indices...,OtherIndices...> res;
    res.set(std::tuple_cat( std::tuple<Indices...>(*this) , std::tuple<OtherIndices...>(other) ));
    return res;
  }
  
  template <size_t B,size_t E,typename Enable = void> struct make_slice;
  template <size_t B,size_t E> struct make_slice<B,E,typename std::enable_if<B == E>::type>{ using type = index_tuple<>; };
  template <size_t B,size_t E> struct make_slice<B,E,typename std::enable_if<B < E>::type>{ using type = typename index_tuple<element_type<B>>::template append_type<typename make_slice<B+1,E>::type>; };
  template <size_t B,size_t E> using slice_type = typename make_slice<B, E>::type;
  
  template <size_t Begin, size_t End, typename Args> void set_from_arg_range(const Args &args){
    static_assert(End - Begin == size(), "invalid range");
    setter<Args,Begin> value_setter(args);
    apply_template(value_setter);
  }
  
  template <size_t B,size_t E> slice_type<B,E> slice()const{
    typename make_slice<B,E>::type res;
    res.template set_from_arg_range<B,E>( std::tuple<Indices...>(*this) );
    return res;
  }

  template<typename I> using push_back_type = index_tuple<Indices..., I >;
  template<size_t N> using push_back_static_type = index_tuple<Indices..., static_index<N> >;
  using push_back_dynamic_type = index_tuple<Indices..., dynamic_index>;
  
  template<typename I> using push_front_type = index_tuple< I, Indices... >;
  template<size_t N> using push_front_static_type = index_tuple<static_index<N>, Indices... >;
  using push_front_dynamic_type = index_tuple<dynamic_index,Indices...>;
  
  template <size_t value> push_back_static_type<value> push_back(){ return append(index_tuple<static_index<1>>()); }
  push_back_dynamic_type push_back(size_t value){ return append(index_tuple<dynamic_index>(value)); }
  
  template <size_t value> push_front_static_type<value> push_front(){ return index_tuple<static_index<1>>().append(*this); }
  push_front_dynamic_type push_front(size_t value){ return index_tuple<dynamic_index>(value).append(*this); }
  
  template <class Red, typename Other> using reduced_result = typename reducer<index_tuple<Indices...>,Other,Red>::result_type;
  
  template <typename Other> using sum_result = reduced_result<sum_reduce, Other>;
  template <typename Other> using mul_result = reduced_result<mul_reduce, Other>;
  template <typename Other> using dif_result = reduced_result<dif_reduce, Other>;
  template <typename Other> using div_result = reduced_result<div_reduce, Other>;
  
  template <typename ... OtherIndices> sum_result<index_tuple<OtherIndices...>> operator+(const index_tuple<OtherIndices...> &other)const{
    return reduce<sum_reduce>(other);
  }
  
  template <typename ... OtherIndices> dif_result<index_tuple<OtherIndices...>> operator-(const index_tuple<OtherIndices...> &other)const{
    return reduce<dif_reduce>(other);
  }
  
  template <typename ... OtherIndices> mul_result<index_tuple<OtherIndices...>> operator*(const index_tuple<OtherIndices...> &other)const{
    return reduce<mul_reduce>(other);
  }
  
  template <typename ... OtherIndices> div_result<index_tuple<OtherIndices...>> operator/(const index_tuple<OtherIndices...> &other)const{
    return reduce<div_reduce>(other);
  }
  
  };
  
  template <size_t ... Indices> using static_index_tuple = index_tuple<static_index<Indices> ...>;
  
  template <size_t N> struct make_dynamic_index_tuple_type{ using type = typename make_dynamic_index_tuple_type<N-1>::type::push_back_dynamic_type; };
  template <> struct make_dynamic_index_tuple_type<0>{ using type = index_tuple<>; };
  template <size_t N> using dynamic_index_tuple = typename make_dynamic_index_tuple_type<N>::type;
  template <typename ... Args> dynamic_index_tuple<sizeof...(Args)> make_dynamic_index_tuple(Args ... args){ return dynamic_index_tuple<sizeof...(Args)>(args...); }
  
  
  template <typename ... Indices> std::ostream & operator<<(std::ostream &stream, const index_tuple<Indices...> & idx){
    stream << '(';
    idx.apply([&](size_t i,size_t val){ stream << val; if(i+1 != idx.size()) stream << ','; });
    stream << ')';
    return stream;
  }
  
  template <size_t N> struct make_index_tuple_range{ using type = typename make_index_tuple_range<N-1>::type::template make_append<index_tuple<static_index<N-1>>>::type; };
  template <> struct make_index_tuple_range<0>{ using type = index_tuple<>; };
  template <size_t N> using index_tuple_range = typename make_index_tuple_range<N>::type;
  
  template <typename IndexTuple,size_t N = IndexTuple::size()> struct make_reverse_index_tuple{
    using current = index_tuple<typename IndexTuple::template element_type<N-1>>;
    using rest = typename make_reverse_index_tuple<IndexTuple,N-1>::type;
    using type = typename current::template make_append<rest>::type;
    static type reverse(IndexTuple tuple){ return current(tuple.template get<N-1>()).append(make_reverse_index_tuple<IndexTuple,N-1>::reverse(tuple)); }
  };
  template <typename IndexTuple> struct make_reverse_index_tuple<IndexTuple,0>{ using type = index_tuple<>; static type reverse(IndexTuple tuple){ return type(); } };
  template <typename IndexTuple> using reversed_index_tuple_type = typename make_reverse_index_tuple<IndexTuple>::type;
  template <typename IndexTuple> reversed_index_tuple_type<IndexTuple> reverse(IndexTuple tuple){ return make_reverse_index_tuple<IndexTuple>::reverse(tuple); }
  
  template <size_t V,size_t N> struct make_index_tuple_constant{ using type = typename make_index_tuple_constant<V,N-1>::type::template make_append<index_tuple<static_index<V>>>::type; };
  template <size_t V> struct make_index_tuple_constant<V,0>{ using type = index_tuple<>; };
  template <size_t V,size_t N> using index_tuple_constant = typename make_index_tuple_constant<V,N>::type;

  
  
}

