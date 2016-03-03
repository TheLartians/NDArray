#include "iterators.h"
#include "index_tuple.h"
#include <array>

namespace lars{

template <class Shape,size_t N = Shape::size()> struct ndarray_calculator{
  using next_prod_type = typename ndarray_calculator<Shape,N-1>::next_prod_type::template slice_type<1,Shape::size()>::template push_back_static_type<1>;
  using prod_type = typename next_prod_type::template mul_result< typename ndarray_calculator<Shape,N-1>::prod_type  >;
  using stride_type = typename prod_type::template slice_type<1,Shape::size()>::template push_back_static_type<1>;
  static next_prod_type next_prod(const Shape &shape){ return ndarray_calculator<Shape,N-1>::next_prod(shape).template slice<1,Shape::size()>().append(static_index_tuple<1>()); }
  static prod_type prod(const Shape &shape){ return next_prod(shape) * ndarray_calculator<Shape,N-1>::prod(shape); }
  static stride_type prod_to_stride(const prod_type &prod){ return prod.template slice<1,prod_type::size()>().template push_back<1>(); }
  static stride_type stride(const Shape &shape){ return prod_to_stride(prod(shape)); }
};

template <class Shape> struct ndarray_calculator<Shape,0>{
  using next_prod_type = Shape;
  using prod_type = Shape;
  using stride_type = Shape;
  static next_prod_type next_prod(const Shape &shape){ return shape; }
  static prod_type prod(const Shape &shape){ return shape; }
  static stride_type prod_to_stride(const prod_type &prod){ return prod.template slice<1,prod_type::size()>().template push_back<1>(); }
  static stride_type stride(const Shape &shape){ return prod_to_stride(prod(shape)); }
};

template <class T> struct borrowed_data{
  T * data = nullptr;
  T * get()const{ return data; }
  borrowed_data(T * source):data(source){ }
};

template <class T,typename Shape,typename Stride,typename Offset,typename Data> class ndarray_base{
  
public:
  
  Shape _shape;
  Stride _stride;
  Offset _offset;
  
  Data data;
  
  struct check_index{
    Shape shape;
    check_index(Shape _shape):shape(_shape){}
    
    template <size_t Idx> void operator()(const dynamic_index &idx)const{
      if(idx >= shape.template get<Idx>()) throw std::range_error("invalid array index " + std::to_string(Idx) + ": " + std::to_string(idx));
    }
    
    template <size_t Idx,size_t value> typename std::enable_if<!Shape::template element_type<Idx>::is_dynamic>::type operator()(const static_index<value> &v)const{
      static_assert( value < Shape::template get<Idx>(), "invalid array index" );
    }
    
    template <size_t Idx,size_t value> typename std::enable_if<Shape::template element_type<Idx>::is_dynamic>::type operator()(const static_index<value> &v)const{
      if(value >= shape.template get<Idx>()) throw std::range_error("invalid array index " + std::to_string(Idx) + ": " + std::to_string(value));
    }
    
  };
  
  
public:
  
  using index_type = dynamic_index_tuple<Shape::size()>;
  
  template <typename ... DataArgs> ndarray_base(Shape shape,Stride stride,Offset offset,DataArgs ... data):_shape(shape),_stride(stride),_offset(offset),data(data...){ }
  
  ndarray_base(const ndarray_base &other) = delete;
  ndarray_base(ndarray_base &&other) = default;
  
  static constexpr size_t ndim(){ return Shape::size(); }
  size_t offset()const{ return _offset; }
  constexpr size_t size()const{ return _shape.template get<0>(); }
  const Shape & shape()const{ return _shape; }
  const Stride &stride()const{ return _stride; }
  
  template <typename Index> size_t get_data_index(Index idx){
#ifndef NDEBUG
    check_index check(shape());
    idx.apply_template(check);
#endif
    size_t i = offset();
    (_stride * idx).apply([&](size_t idx,size_t v){ i+=v; });
    return i;
  }
  
  template <typename ... Args> T & operator()(Args ... args){
    return data.get()[get_data_index(index_type(args...))];
  }
  
  template <typename Pos,typename Sha,typename Ste> using slice_type = ndarray_base<T,Sha, typename Stride::template mul_result<Ste>, dynamic_index, borrowed_data<T> >;
  
  template <typename Pos,typename Sha,typename Ste = index_tuple_constant<1, Shape::size()>> slice_type<Pos,Sha,Ste> slice(Pos pos,Sha shape,Ste step = index_tuple_constant<1, Shape::size()>()){
#ifndef  NDEBUG
    check_index check(_shape);
    (pos + step*(shape - index_tuple_constant<1, Shape::size()>())).apply_template(check);
#endif
    size_t offset = get_data_index(pos);
    return slice_type<Pos,Sha,Ste>(shape,stride()*step,offset,data.get());
  }
  
  using element_type = ndarray_base<T,typename Shape::template slice_type<1,Shape::size()>,typename Stride::template slice_type<1,Shape::size()>, dynamic_index, borrowed_data<T> >;
  using const_element_type = ndarray_base<T,typename Shape::template slice_type<1,Shape::size()>,typename Stride::template slice_type<1,Shape::size()>, dynamic_index, borrowed_data<const T> >;
  
  template <typename D> struct dummy_template{ const static bool value = true; };
  template <typename Ret,typename Dummy> using enable_if_one_dimensional = typename std::enable_if<(Shape::size() == 1) && dummy_template<Dummy>::value,Ret>::type;
  template <typename Ret,typename Dummy> using disable_if_one_dimensional = typename std::enable_if<(Shape::size() > 1) && dummy_template<Dummy>::value,Ret>::type;

  struct iterator:public std::iterator<std::input_iterator_tag, element_type>{
    ndarray_base & parent;
    size_t current_index;
    iterator(ndarray_base &_parent,size_t index):parent(_parent),current_index(index){ }
    element_type operator*()const{ return parent[current_index]; }
    iterator & operator++(){ ++current_index; return *this; }
    iterator operator++()const{return iterator(parent, current_index+1); }
    bool operator!=(const iterator &other)const{ return other.current_index != current_index || other.parent != parent; }
  };
  
  struct const_iterator:public std::iterator<std::input_iterator_tag, const_element_type>{
    const ndarray_base & parent;
    size_t current_index;
    const_iterator(const ndarray_base &_parent,size_t index):parent(_parent),current_index(index){ }
    const_element_type operator*()const{ return parent[current_index]; }
    const_iterator & operator++(){ ++current_index; return *this; }
    const_iterator operator++()const{return const_iterator(parent, current_index+1); }
    bool operator!=(const const_iterator &other)const{ return other.current_index != current_index || other.parent != parent; }
  };
  
  iterator begin(){ return iterator(*this, 0); }
  iterator end(){ return iterator(*this, size()); }
  const_iterator begin()const{ return const_iterator(*this, 0); }
  const_iterator end()const{ return const_iterator(*this, size()); }
  
  template <typename Idx> disable_if_one_dimensional<const const_element_type,Idx> operator[](Idx i)const{
    auto off = offset() + i*stride().template get<0>();
#ifndef NDEBUG
    if(i>=shape().template get<0>()) throw std::range_error("invalid array index");
#endif
    return const_element_type(shape().template slice<1,Shape::size()>(),stride().template slice<1,Shape::size()>(),off,data.get());
  }
  
  template <typename Idx> enable_if_one_dimensional<const T &,Idx> operator[](Idx i)const{
    auto off = offset() + i*stride().template get<0>();
#ifndef NDEBUG
    if(i>=shape().template get<0>()) throw std::range_error("invalid array index");
#endif
    return data.get()[off];
  }
  
  template <typename Idx> disable_if_one_dimensional<element_type,Idx> operator[](Idx i){
    auto off = offset() + i*stride().template get<0>();
#ifndef NDEBUG
    if(i>=shape().template get<0>()) throw std::range_error("invalid array index");
#endif
    return element_type(shape().template slice<1,Shape::size()>(),stride().template slice<1,Shape::size()>(),off,data.get());
  }
  
  template <typename Idx> enable_if_one_dimensional<T &,Idx> operator[](Idx i){
    auto off = offset() + i*stride().template get<0>();
#ifndef NDEBUG
    if(i>=shape().template get<0>()) throw std::range_error("invalid array index");
#endif
    return data.get()[off];
  }
  
  ndarray_base & operator=(const T &value){
    for(auto i:range(size())) (*this)[i] = value;
    return *this;
  }
  
  ndarray_base & operator=(const ndarray_base &other){ return operator=<>(other); }
  ndarray_base & operator=(ndarray_base &&other){ return operator=<>(other); }
  
  template <typename oT,typename oShape,typename oStride,typename oOffset,typename oData> typename std::enable_if<oShape::size() == Shape::size(),ndarray_base&>::type operator=(const ndarray_base<oT, oShape, oStride, oOffset, oData> & other){
#ifndef NDEBUG
    if(other.size() != size()) throw std::runtime_error("invalid assignment");
#endif
    for(auto i:range(size())) (*this)[i] = other[i];
    return *this;
  }
  
  void fill(const T &value){
    *this = value;
  }
  
  template <typename F,typename Idx> enable_if_one_dimensional<void,Idx> element_wise_helper(F f,Idx idx){
    auto i = idx.push_back(0);
    for(auto j:range(size())){
      i.template set<Idx::size()>(j);
      (*this)[j] = f(i);
    }
  }
  
  template <typename F,typename Idx> disable_if_one_dimensional<void,Idx> element_wise_helper(F f,Idx idx){
    auto i = idx.push_back(0);
    for(auto j:range(size())){
      i.template set<Idx::size()>(j);
      (*this)[j].element_wise_helper(f,i);
    }
  }
  
  using transposed_type = ndarray_base<T, reversed_index_tuple_type<Shape> , reversed_index_tuple_type<Stride>, Offset, borrowed_data<T>>;
  using const_transposed_type = ndarray_base<T, reversed_index_tuple_type<Shape> , reversed_index_tuple_type<Stride>, Offset, borrowed_data<const T>>;
  
  transposed_type transpose(){
    return transposed_type(reverse(_shape),reverse(_stride),_offset,data.get());
  }

  const_transposed_type transpose()const{
    return const_transposed_type(reverse(_shape),reverse(_stride),_offset,data.get());
  }
  
  template <typename F> void element_wise(F f){
    element_wise_helper(f, index_tuple<>());
  }
  
};

template <class T,typename Shape,typename Stride,typename Offset,typename Data> std::ostream & operator<<(std::ostream &stream,const ndarray_base<T, Shape, Stride, Offset, Data> & array){
  stream << '[';
  for(auto i:range(array.size()-1)){
    stream << array[i] << ',';
    for(auto i UNUSED:range(array.ndim()-1)) stream << '\n';
  }
  if(array.size() != 0){
    stream << array[array.size()-1];
  }
  stream << ']';
  return stream;
}
  
  template <class T> struct heap_data{
    T * data = nullptr;
    size_t size = 0;
    
    heap_data(size_t size){ resize(size); }
    heap_data(const heap_data &other){ resize(other.size); for(auto i:range(size)) data[i] = other.data[i]; }
    
    heap_data(heap_data && other){ data = other.data; size = other.size; other.size = 0; other.data = nullptr; }
    heap_data &operator=(heap_data && other){ data = other.data; size = other.size; other.size = 0; other.data = nullptr; }
    
    T * get()const{ return data; }
    void resize(size_t _size){ size = _size; if(data){ delete [] data; data = nullptr; } if(size > 0) data = new T[size]; }
    heap_data & operator=(const heap_data &other){ resize(other.size); for(auto i:range(size)) data[i] = other.data[i]; }
    ~heap_data(){ if(data) delete [] data; }
  };
  
  template <class T,size_t size> struct stack_data{
    std::array<T,size> data;
    T * get(){ return &data[0]; }
    const T * get()const{ return &data[0]; }
  };

template <class T,typename Shape> class heap_ndarray:public ndarray_base<T,Shape, typename ndarray_calculator<Shape>::stride_type, static_index<0>,heap_data<T>>{
public:
  
  using base = ndarray_base<T,Shape, typename ndarray_calculator<Shape>::stride_type, static_index<0>,heap_data<T>>;
  using base::operator=;
  
  heap_ndarray(Shape shape = Shape()):base(shape,ndarray_calculator<Shape>::stride(shape),static_index<0>(),ndarray_calculator<Shape>::prod(shape).template get<0>()){}
  
  heap_ndarray(heap_ndarray && other) = default;
  heap_ndarray &operator=(heap_ndarray && other) = default;
};
  
  template <class T,typename Shape> class stack_ndarray:public ndarray_base<T,Shape, typename ndarray_calculator<Shape>::stride_type, static_index<0>,stack_data<T, ndarray_calculator<Shape>::prod_type::template get<0>() >>{
  public:
    
    using base = ndarray_base<T,Shape, typename ndarray_calculator<Shape>::stride_type, static_index<0>,stack_data<T, ndarray_calculator<Shape>::prod_type::template get<0>() >>;
    using base::operator=;
    
    stack_ndarray(Shape shape = Shape()):base(shape,ndarray_calculator<Shape>::stride(shape),static_index<0>()){}
  };

}
