
#pragma once

#include <tuple>
#include <array>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <ostream>
#include <lars/unused.h>

//#define ZERO_SIZE_STATIC_INDEX

namespace lars {
  
struct DynamicIndex{
  size_t value = 0;
  constexpr DynamicIndex(){}
  constexpr DynamicIndex(size_t _value):value(_value){}
  operator size_t()const{ return value; }
  operator size_t&(){ return value; }
  constexpr static bool is_dynamic = true;
};
  
template <size_t _value> struct StaticIndex{
#ifdef ZERO_SIZE_STATIC_INDEX
  char MAKE_SIZE_ZERO_IF_EMPTY[0];
  operator size_t()const{ (void)(MAKE_SIZE_ZERO_IF_EMPTY[0]); /* to silence compiler */ return _value;  }
#else
  operator size_t()const{ return _value;  }
#endif
  constexpr StaticIndex(){ }
  static const size_t value = _value;
  constexpr static bool is_dynamic = false;
};

template <typename ... Indices> class IndexTuple;

template <typename Lhs,typename Rhs,typename F> struct Reducer:public F{
  const Lhs & lhs;
  const Rhs & rhs;
  
  Reducer(const Lhs & _lhs,const Rhs & _rhs):lhs(_lhs),rhs(_rhs){ static_assert(Lhs::size() == Rhs::size(), "index tuple size doesn't match"); }
  
  template <size_t N,typename Enable = void> struct make_ResultType;
  
  template <size_t N> struct make_ResultType<N,typename std::enable_if<N!=0 && (Lhs::template element_is_dynamic<N-1>() || Rhs::template element_is_dynamic<N-1>())>::type>{
    using type = typename make_ResultType<N-1>::type::PushBackDynamic;
  };
  
  template <size_t N> struct make_ResultType<N,typename std::enable_if<N!=0 && !(Lhs::template element_is_dynamic<N-1>() || Rhs::template element_is_dynamic<N-1>())>::type>{
	    using type = typename make_ResultType<N-1>::type::template PushBackStatic<F::reduce(Lhs::template get<N-1>(),Rhs::template get<N-1>())>;
	  };
	  
	  template <size_t N> struct make_ResultType<N,typename std::enable_if<N==0>::type>{ using type = IndexTuple<>; };
	  
	  using ResultType = typename make_ResultType<Lhs::size()>::type;
	  
	  template <size_t Idx> void operator()(DynamicIndex & value){
	    value = F::reduce(lhs.template get<Idx>(),rhs.template get<Idx>());
	  }
	  
	  template <size_t Idx,size_t value> void operator()(StaticIndex<value> &){  }
	  
	};


	template <typename ... Indices> class IndexTuple:public std::tuple<Indices...>{

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
	  
	  template <size_t Idx> using ElementType = typename std::tuple_element<Idx, std::tuple<Indices...> >::type;
	  
	  template <size_t Idx> constexpr static typename std::enable_if<(Idx<size()),bool>::type element_is_dynamic(){ return std::tuple_element<Idx, std::tuple<Indices...> >::type::is_dynamic; }
	  
	  template <size_t i=size()-1> constexpr static typename std::enable_if<i != 0,bool>::type is_dynamic(){ return element_is_dynamic<i>() || is_dynamic<i-1>(); }
	  template <size_t i=size()-1> constexpr static typename std::enable_if<i == 0,bool>::type is_dynamic(){ return element_is_dynamic<i>(); }
	  
	private:
	  
	  struct mul_reduce{ constexpr static size_t reduce(size_t a,size_t b){ return a*b; } };
	  struct sum_reduce{ constexpr static size_t reduce(size_t a,size_t b){ return a+b; } };
	  struct div_reduce{ constexpr static size_t reduce(size_t a,size_t b){ return a/b; } };
	  struct dif_reduce{ constexpr static size_t reduce(size_t a,size_t b){ return a-b; } };
	  
	  template <typename Args,size_t Begin = 0> struct Setter{
	    const Args & values;
	    Setter(const Args &_values):values(_values){}
	    
	    template <size_t Idx> using enable_if_dynamic = typename std::enable_if<std::tuple_element<Idx, Args>::type::is_dynamic>::type;
	    template <size_t Idx> using enable_if_not_dynamic = typename std::enable_if<!std::tuple_element<Idx, Args>::type::is_dynamic>::type;
	    
	    template <size_t Idx>  void operator()(DynamicIndex & value)const{ value = size_t(std::get<Idx + Begin>(values)); }
	    
	    template <size_t Idx,size_t value> enable_if_dynamic<Idx> operator()(StaticIndex<value> &v)const{
	#ifndef NDEBUG
	      if(std::get<Idx + Begin>(values) != value) throw std::runtime_error("changing static index");
	#endif
	    }
	    
	    template <size_t Idx,size_t value> enable_if_not_dynamic<Idx> operator()(StaticIndex<value> &v)const{
	      static_assert( std::tuple_element<Idx + Begin, Args>::type::value == value, "changing static index");
	    }

	  };
	  
	  template <size_t Idx,class Ret> using enable_if_dynamic = typename std::enable_if<std::tuple_element<Idx, std::tuple<Indices...>>::type::is_dynamic,Ret>::type;
	  template <size_t Idx,class Ret> using enable_if_not_dynamic = typename std::enable_if<(!std::tuple_element<Idx, std::tuple<Indices...>>::type::is_dynamic),Ret>::type;

	public:
	  
	  IndexTuple(){}
	  
	  template <typename ... Args> IndexTuple(const IndexTuple<Args...> &other){ set(other); }
	  template <typename ... Args> IndexTuple(Args ... args){ set(args...); }
	  
	  template <typename ... Args> void set(const std::tuple<Args...> &args){
	    static_assert(sizeof...(Args) == size(), "index tuple size doesn't match size");
	    Setter<std::tuple<Args...>> value_setter(args);
	    apply_template(value_setter);
	  }

	  template <typename ... Args> IndexTuple & operator=(const IndexTuple<Args...> &other){
	    set((const std::tuple<Args...> &)other);
	    return *this;
	  }
	  
	  template <typename F,typename Rhs> typename Reducer<IndexTuple<Indices...>,Rhs,F>::ResultType reduce(Rhs rhs)const{
	    typename Reducer<IndexTuple<Indices...>,Rhs,F>::ResultType result;
	    Reducer<IndexTuple<Indices...>,Rhs,F> Reducer(*this,rhs);
	    result.apply_template(Reducer);
	    return result;
	  }
	  
	  template <typename ... Args> void set(size_t first, Args ... args){ set(std::make_tuple( DynamicIndex(first), DynamicIndex(args) ...) ); }
	  
	  template <size_t Idx> void set(size_t value){ std::get<Idx>(*this) = value; }
	  template <size_t ... Idx> void set(){ set(std::make_tuple( StaticIndex<Idx>() ... ) ); }
	  
	  template <size_t Idx> enable_if_dynamic<Idx, size_t> get()const{ return std::get<Idx>(*this).value; }
	  template <size_t Idx> enable_if_not_dynamic<Idx, size_t> static constexpr get(){ return std::tuple_element<Idx, std::tuple<Indices...> >::type::value; }


	  template <size_t Idx> struct ValidGetter{ const size_t value = IndexTuple::get<Idx>(); };
	  template <size_t Idx> struct OutOfRangeGetter{ const size_t value = 0; };
	  
	  template <typename> struct make_append;
	  template <typename... OtherIndices> struct make_append<IndexTuple<OtherIndices...>> { using type = IndexTuple<Indices..., OtherIndices...>; };
	  template <typename Other> using Append = typename make_append<Other>::type;
	  
	  template <typename... OtherIndices> IndexTuple<Indices...,OtherIndices...> append(const IndexTuple<OtherIndices...> &other)const{
	    IndexTuple<Indices...,OtherIndices...> res;
	    res.set(std::tuple_cat( std::tuple<Indices...>(*this) , std::tuple<OtherIndices...>(other) ));
	    return res;
	  }
	  
	  template <size_t B,size_t E,typename Enable = void> struct make_slice;
	  template <size_t B,size_t E> struct make_slice<B,E,typename std::enable_if<B == E>::type>{ using type = IndexTuple<>; };
	  template <size_t B,size_t E> struct make_slice<B,E,typename std::enable_if<B < E>::type>{ using type = typename IndexTuple<ElementType<B>>::template Append<typename make_slice<B+1,E>::type>; };
	  template <size_t B,size_t E> using Slice = typename make_slice<B, E>::type;
	  
	  template <size_t Begin, size_t End, typename Args> void set_from_arg_range(const Args &args){
	    static_assert(End - Begin == size(), "invalid range");
	    Setter<Args,Begin> value_setter(args);
	    apply_template(value_setter);
	  }
	  
	  template <size_t B,size_t E> Slice<B,E> slice()const{
	    typename make_slice<B,E>::type res;
	    res.template set_from_arg_range<B,E>( std::tuple<Indices...>(*this) );
	    return res;
	  }

	  template<typename I> using PushBack = IndexTuple<Indices..., I >;
	  template<size_t N> using PushBackStatic = IndexTuple<Indices..., StaticIndex<N> >;
	  using PushBackDynamic = IndexTuple<Indices..., DynamicIndex>;
	  
	  template<typename I> using PushFront = IndexTuple< I, Indices... >;
	  template<size_t N> using PushFrontStatic = IndexTuple<StaticIndex<N>, Indices... >;
	  using PushFrontDynamic = IndexTuple<DynamicIndex,Indices...>;
	  
	  template <size_t value> PushBackStatic<value> push_back(){ return append(IndexTuple<StaticIndex<1>>()); }
	  PushBackDynamic push_back(size_t value){ return append(IndexTuple<DynamicIndex>(value)); }
	  
	  template <size_t value> PushFrontStatic<value> push_front(){ return IndexTuple<StaticIndex<1>>().append(*this); }
	  PushFrontDynamic push_front(size_t value){ return IndexTuple<DynamicIndex>(value).append(*this); }
	  
	  template <class Red, typename Other> using reduced_result = typename Reducer<IndexTuple<Indices...>,Other,Red>::ResultType;
	  
	  template <typename Other> using sum_result = reduced_result<sum_reduce, Other>;
	  template <typename Other> using mul_result = reduced_result<mul_reduce, Other>;
	  template <typename Other> using dif_result = reduced_result<dif_reduce, Other>;
	  template <typename Other> using div_result = reduced_result<div_reduce, Other>;
	  
	  
	  template <size_t Idx,typename ... OtherIndices> typename std::enable_if<Idx != 0,bool>::type is_equal(const IndexTuple<OtherIndices...> &other)const{
	    return get<Idx>() == other.template get<Idx>() && is_equal<Idx-1>(other);
	  }
	  
	  template <size_t Idx,typename ... OtherIndices> typename std::enable_if<Idx == 0,bool>::type is_equal(const IndexTuple<OtherIndices...> &other)const{
	    return get<Idx>() == other.template get<Idx>();
	  }
	  
	  template <typename ... OtherIndices> bool operator==(const IndexTuple<OtherIndices...> &other)const{
	    return is_equal<size()-1>(other);
	  }
	  
	  template <typename ... OtherIndices> bool operator!=(const IndexTuple<OtherIndices...> &other)const{
	    return !is_equal<size()-1>(other);
	  }
	  
	  template <typename ... OtherIndices> sum_result<IndexTuple<OtherIndices...>> operator+(const IndexTuple<OtherIndices...> &other)const{
	    return reduce<sum_reduce>(other);
	  }
	  
	  template <typename ... OtherIndices> dif_result<IndexTuple<OtherIndices...>> operator-(const IndexTuple<OtherIndices...> &other)const{
	    return reduce<dif_reduce>(other);
	  }
	  
	  template <typename ... OtherIndices> mul_result<IndexTuple<OtherIndices...>> operator*(const IndexTuple<OtherIndices...> &other)const{
	    return reduce<mul_reduce>(other);
	  }
	  
	  template <typename ... OtherIndices> div_result<IndexTuple<OtherIndices...>> operator/(const IndexTuple<OtherIndices...> &other)const{
	    return reduce<div_reduce>(other);
	  }
	  
	  template <size_t Idx> struct IndexIfInvalid{ template <size_t I2> static constexpr size_t get(){ return 0; } };
	  template <size_t Idx> struct IndexIfValid{
	    template <size_t I2> static constexpr enable_if_not_dynamic<I2, size_t> get(){ return IndexTuple::get<Idx>(); }
	    template <size_t I2> static constexpr enable_if_dynamic<I2, size_t> get(){ return 0; }
	  };
	  
	  template <size_t Idx> constexpr static size_t safe_static_get(){
	    return std::conditional<Idx < size(), IndexIfValid<Idx>, IndexIfInvalid<Idx>>::type::template get<Idx>();
	  }
	  
	  template <size_t Idx> typename std::enable_if<Idx < size(),size_t>::type safe_get()const{
	    return get<Idx>();
	  }
	  
	  template <size_t Idx> typename std::enable_if<Idx >= size(),size_t>::type safe_get()const{
	    return 0;
	  }
	  
	  };
	  
	  template <size_t ... Indices> using StaticIndexTuple = IndexTuple<StaticIndex<Indices> ...>;
	  
	  template <size_t N> struct make_dynamic_index_tuple_type{ using type = typename make_dynamic_index_tuple_type<N-1>::type::PushBackDynamic; };
	  template <> struct make_dynamic_index_tuple_type<0>{ using type = IndexTuple<>; };
	  template <size_t N> using DynamicIndexTuple = typename make_dynamic_index_tuple_type<N>::type;
	  template <typename ... Args> DynamicIndexTuple<sizeof...(Args)> make_dynamic_index_tuple(Args ... args){ return DynamicIndexTuple<sizeof...(Args)>(args...); }
	  
	  template <typename ... Indices> std::ostream & operator<<(std::ostream &stream, const IndexTuple<Indices...> & idx){
	    stream << '(';
	    idx.apply([&](size_t i,size_t val){ stream << val; if(i+1 != idx.size()) stream << ','; });
    stream << ')';
    return stream;
  }
  
  template <size_t N> struct make_index_tuple_range{ using type = typename make_index_tuple_range<N-1>::type::template make_append<IndexTuple<StaticIndex<N-1>>>::type; };
  template <> struct make_index_tuple_range<0>{ using type = IndexTuple<>; };
  template <size_t N> using IndexTupleRange = typename make_index_tuple_range<N>::type;
  
  template <typename IndexTupleT,size_t N = IndexTupleT::size()> struct make_reverse_index_tuple{
    using current = IndexTuple<typename IndexTupleT::template ElementType<N-1>>;
    using rest = typename make_reverse_index_tuple<IndexTupleT,N-1>::type;
    using type = typename current::template make_append<rest>::type;
    static type reverse(IndexTupleT tuple){ return current(tuple.template get<N-1>()).append(make_reverse_index_tuple<IndexTupleT,N-1>::reverse(tuple)); }
  };
  template <typename IndexTupleT> struct make_reverse_index_tuple<IndexTupleT,0>{ using type = IndexTuple<>; static type reverse(IndexTupleT tuple){ return type(); } };
  template <typename IndexTupleT> using ReversedIndexTuple = typename make_reverse_index_tuple<IndexTupleT>::type;
  template <typename IndexTupleT> ReversedIndexTuple<IndexTupleT> reverse(IndexTupleT tuple){ return make_reverse_index_tuple<IndexTupleT>::reverse(tuple); }
  
  template <size_t V,size_t N> struct make_index_tuple_repeat{ using type = typename make_index_tuple_repeat<V,N-1>::type::template make_append<IndexTuple<StaticIndex<V>>>::type; };
  template <size_t V> struct make_index_tuple_repeat<V,0>{ using type = IndexTuple<>; };
  template <size_t V,size_t N> using IndexTupleRepeat = typename make_index_tuple_repeat<V,N>::type;

  template <size_t N> using IndexTupleZeros = typename make_index_tuple_repeat<0,N>::type;
  template <size_t N> using IndexTupleOnes = typename make_index_tuple_repeat<1,N>::type;
  
  
  
}

