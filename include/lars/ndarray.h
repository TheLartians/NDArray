#pragma once

#include <lars/iterators.h>
#include <lars/index_tuple.h>
#include <lars/dummy.h>
#include <lars/to_string.h>

#include <array>
#include <vector>
#include <initializer_list>
#include <type_traits>
#include <cmath>

namespace lars{
  
  
  template <class Shape,size_t N = Shape::size()> struct NDArrayCalculator{
    using NextProd = typename NDArrayCalculator<Shape,N-1>::NextProd::template Slice<1,Shape::size()>::template PushBackStatic<1>;
    using Prod = typename NextProd::template mul_result< typename NDArrayCalculator<Shape,N-1>::Prod  >;
    using Stride = typename Prod::template Slice<1,Shape::size()>::template PushBackStatic<1>;
    using Size = typename Prod::template ElementType<0>;
    
    static NextProd next_prod(const Shape &shape){ return NDArrayCalculator<Shape,N-1>::next_prod(shape).template slice<1,Shape::size()>().append(StaticIndexTuple<1>()); }
    static Prod prod(const Shape &shape){ return next_prod(shape) * NDArrayCalculator<Shape,N-1>::prod(shape); }
    static Stride prod_to_stride(const Prod &prod){ return prod.template slice<1,Prod::size()>().template push_back<1>(); }
    static Stride stride(const Shape &shape){ return prod_to_stride(prod(shape)); }
    static Size size(const Shape &shape){ return prod(shape).template get<0>(); }
  };
  
  template <class Shape> struct NDArrayCalculator<Shape,0>{
    using NextProd = Shape;
    using Prod = Shape;
    using Stride = Shape;
    static NextProd next_prod(const Shape &shape){ return shape; }
    static Prod prod(const Shape &shape){ return shape; }
    static Stride prod_to_stride(const Prod &prod){ return prod.template slice<1,Prod::size()>().template push_back<1>(); }
    static Stride stride(const Shape &shape){ return prod_to_stride(prod(shape)); }
  };
  
  template <class T> struct BorrowedData{
    T * data = nullptr;
    T * get()const{ return data; }
    BorrowedData(T * source):data(source){ }
  };
  
  template <class Shape> class ExplicitShapeStorage{
    Shape _shape;
  public:
    ExplicitShapeStorage(Shape shape = Shape()):_shape(shape){ }
    const Shape & shape()const{ return _shape; }
    void set(const Shape &shape){ _shape.set(shape); }
  };
  
  template <class Stride> class ExplicitStrideStorage{
    Stride _stride;
  public:
    ExplicitStrideStorage(Stride stride = Stride()):_stride(stride){ }
    const Stride &stride()const{ return _stride; }
    void set(const Stride &stride){ _stride.set(stride); }
  };
  
  template <class Offset> class ExplicitOffsetStorage{
    Offset _offset;
  public:
    ExplicitOffsetStorage(Offset offset = Offset()):_offset(offset){ }
    Offset offset()const{ return _offset; }
    void set(const Offset &offset){ _offset.set(offset); }
  };
  
  template <class Shape> struct ImplicitShapeStorage{
    ImplicitShapeStorage(Shape shape = Shape()){}
    Shape shape()const{ return Shape(); }
    void set(const Shape &shape){ static_assert(!Shape::is_dynamic(), "implicit shape cannot be dynamic"); }
  };
  
  template <class Stride> struct ImplicitStrideStorage{
    ImplicitStrideStorage(Stride stride = Stride()){}
    Stride stride()const{ return Stride(); }
    void set(const Stride &stride){ static_assert(!Stride::is_dynamic(), "implicit stride cannot be dynamic"); }
  };
  
  template <class Offset> struct ImplicitOffsetStorage{
    ImplicitOffsetStorage(Offset offset = Offset()){}
    Offset offset()const{ return Offset(); }
    void set(const Offset &offset){ static_assert(!Offset::is_dynamic, "implicit offset cannot be dynamic"); }
  };
  
  template <class Shape> using ShapeStorage = typename std::conditional<Shape::is_dynamic(), ExplicitShapeStorage<Shape>, ImplicitShapeStorage<Shape>>::type;
  template <class Stride> using StrideStorage = typename std::conditional<Stride::is_dynamic(), ExplicitStrideStorage<Stride>, ImplicitStrideStorage<Stride>>::type;
  template <class Offset> using OffsetStorage = typename std::conditional<Offset::is_dynamic, ExplicitOffsetStorage<Offset>, ImplicitOffsetStorage<Offset>>::type;
  
  template <class T,typename TShape,typename TStride,typename TOffset,typename TData,typename TCreator> class NDArrayBase:public ShapeStorage<TShape>,public StrideStorage<TStride>,public OffsetStorage<TOffset>{
  public:
    
    using Shape = TShape;
    using Stride = TStride;
    using Offset = TOffset;
    using Data = TData;
    using Creator = TCreator;
    
    template <class Ty,class Sh, class St, class Of, class Da> using NDArray = typename Creator::template NDArray<Ty,Sh,St,Of,Da>;
    using Copy = typename Creator::template NewNDArray<typename std::remove_const<T>::type,Shape>;
    template <class T2> using CopyWithType = typename Creator::template NewNDArray<T2,Shape>;
    using Super = NDArray<T,Shape,Stride,Offset,Data>;
    
    using Scalar = T;
    
    static Copy create_zeros(Shape s = Shape()){ Copy a(s); a = 0; return a; }
    static Copy create_ones(Shape s = Shape()) { Copy a(s); a = 1; return a; }
    
  protected:
    
    Data _data;
    
    struct IndexCheck{
      Shape shape;
      IndexCheck(Shape _shape):shape(_shape){}
      
      template <size_t Idx> void operator()(const DynamicIndex &idx)const{
        if(idx >= shape.template get<Idx>()) throw std::range_error("invalid array index " + lars::to_string(Idx) + ": " + lars::to_string(idx));
      }
      
      template <size_t Idx,size_t value> typename std::enable_if<!Shape::template ElementType<Idx>::is_dynamic>::type operator()(const StaticIndex<value> &v)const{
        static_assert( value < Shape::template get<Idx>(), "invalid array index" );
      }
      
      template <size_t Idx,size_t value> typename std::enable_if<Shape::template ElementType<Idx>::is_dynamic>::type operator()(const StaticIndex<value> &v)const{
        if(value >= shape.template get<Idx>()) throw std::range_error("invalid array index " + lars::to_string(Idx) + ": " + lars::to_string(value));
      }
      
    };
    
    using SShape = ShapeStorage<TShape>;
    using SStride = StrideStorage<TStride>;
    using SOffset = OffsetStorage<TOffset>;
    
  public:
    
    using SShape::shape;
    using SStride::stride;
    using SOffset::offset;
    
    T * data() { return _data.get(); }
    const T * data()const  { return _data.get(); }
    
    void swap(NDArrayBase &other){
      std::swap(_data,other._data);
      std::swap((SShape&)*this,(SShape&)other);
      std::swap((SStride&)*this,(SStride&)other);
      std::swap((SOffset&)*this,(SOffset&)other);
    }
    
    using Index = DynamicIndexTuple<Shape::size()>;
    template <size_t ... indices> using StaticIndexType = StaticIndexTuple<indices...>;
    
    template <class Ty,class Sh, class St, class Of, class Da, class Cr> NDArrayBase(const NDArrayBase<Ty,Sh,St,Of,Da,Cr> &other){
#ifndef NDEBUG
      if(other.size() != size()) throw std::runtime_error("invalid assignment");
#endif
      for(auto i:range(size())) (*this)[i] = other[i];
    }
    
    NDArrayBase(Shape shape,Stride stride,Offset offset,Data && data):SShape(shape),SStride(stride),SOffset(offset),_data(std::forward<Data>(data)){ }
    NDArrayBase(Shape shape,Stride stride,Offset offset,const Data & data):SShape(shape),SStride(stride),SOffset(offset),_data(data){ }
    
    template <typename ... DataArgs> NDArrayBase(Shape shape,Stride stride,Offset offset,DataArgs ... data_args):SShape(shape),SStride(stride),SOffset(offset),_data(data_args...){ }
    
    NDArrayBase(const NDArrayBase &other) = delete;
    NDArrayBase(NDArrayBase && other) = default;
    
    static constexpr size_t ndim(){ return Shape::size(); }
    
    constexpr size_t size()const{ return shape().template get<0>(); }
    
    template <typename Index> size_t get_data_index(Index idx)const{
#ifndef NDEBUG
      IndexCheck check(shape());
      idx.apply_template(check);
#endif
      size_t i = offset();
      (stride() * idx).apply([&](size_t idx,size_t v){ i+=v; });
      return i;
    }
    
    const T & operator()(const Index &idx)const{
      return data()[get_data_index(idx)];
    }
    
    T & operator()(const Index &idx){
      return data()[get_data_index(idx)];
    }
    
    template <typename Pos,typename Sha,typename Ste= IndexTupleRepeat<1, Shape::size()>> using Slice = NDArray<T,Sha, typename Stride::template mul_result<Ste>, DynamicIndex, BorrowedData<T> >;
    template <typename Pos,typename Sha,typename Ste = IndexTupleRepeat<1, Shape::size()>> using ConstSlice = NDArray<const T,Sha, typename Stride::template mul_result<Ste>, DynamicIndex, BorrowedData<const T> >;
    
    template <typename Pos,typename Sha,typename Ste = IndexTupleRepeat<1, Shape::size()>> Slice<Pos,Sha,Ste> slice(Pos pos,Sha shape,Ste step = IndexTupleRepeat<1, Shape::size()>()){
#ifndef  NDEBUG
      if(shape != IndexTupleRepeat<0, Shape::size()>()){
        IndexCheck check(SShape::shape());
        (pos + step*(shape - IndexTupleRepeat<1, Shape::size()>())).apply_template(check);
      }
#endif
      size_t offset = get_data_index(pos);
      return Slice<Pos,Sha,Ste>(shape,stride()*step,offset,data());
    }
    
    template <typename Pos,typename Sha,typename Ste = IndexTupleRepeat<1, Shape::size()>> ConstSlice<Pos,Sha,Ste> slice(Pos pos,Sha shape,Ste step = IndexTupleRepeat<1, Shape::size()>())const{
#ifndef  NDEBUG
      if(shape != IndexTupleRepeat<0, Shape::size()>()){
        IndexCheck check(SShape::shape());
        (pos + step*(shape - IndexTupleRepeat<1, Shape::size()>())).apply_template(check);
      }
#endif
      size_t offset = get_data_index(pos);
      return ConstSlice<Pos,Sha,Ste>(shape,stride()*step,offset,data());
    }
    
    using ElementType = typename std::conditional<( Shape::size() > 1), NDArray<T,typename Shape::template Slice<1,Shape::size()>,typename Stride::template Slice<1,Shape::size()>, DynamicIndex, BorrowedData<T> >, T & >::type;
    
    using ConstElementType = typename std::conditional<( Shape::size() > 1), NDArray<const T,typename Shape::template Slice<1,Shape::size()>,typename Stride::template Slice<1,Shape::size()>, DynamicIndex, BorrowedData<const T> >, const T & >::type;
    
    template <typename Ret,typename Dummy> using enable_if_one_dimensional = typename std::enable_if<(Shape::size() == 1) && DummyTemplate<Dummy>::value,Ret>::type;
    template <typename Ret,typename Dummy> using disable_if_one_dimensional = typename std::enable_if<(Shape::size() > 1) && DummyTemplate<Dummy>::value,Ret>::type;
    
    struct iterator:public std::iterator<std::input_iterator_tag, std::remove_reference<ElementType> >{
      NDArrayBase & parent;
      size_t current_index;
      iterator(NDArrayBase &_parent,size_t index):parent(_parent),current_index(index){ }
      ElementType operator*()const{ return parent[current_index]; }
      iterator & operator++(){ ++current_index; return *this; }
      iterator operator++()const{return iterator(parent, current_index+1); }
      bool operator!=(const iterator &other)const{ return other.current_index != current_index || &other.parent != &parent; }
    };
    
    struct const_iterator:public std::iterator<std::input_iterator_tag, std::remove_reference<ConstElementType> >{
      const NDArrayBase & parent;
      size_t current_index;
      const_iterator(const NDArrayBase &_parent,size_t index):parent(_parent),current_index(index){ }
      ConstElementType operator*()const{ return std::move(parent[current_index]); }
      const_iterator & operator++(){ ++current_index; return *this; }
      const_iterator operator++()const{return const_iterator(parent, current_index+1); }
      bool operator!=(const const_iterator &other)const{ return other.current_index != current_index || &other.parent != &parent; }
    };
    
    iterator begin(){ return iterator(*this, 0); }
    iterator end(){ return iterator(*this, size()); }
    const_iterator begin()const{ return const_iterator(*this, 0); }
    const_iterator end()const{ return const_iterator(*this, size()); }
    
    template <typename D = void> disable_if_one_dimensional<ConstElementType,D> operator[](size_t i)const{
      auto off = offset() + i*stride().template get<0>();
#ifndef NDEBUG
      if(i>=shape().template get<0>()) throw std::range_error("invalid array index");
#endif
      return ConstElementType(shape().template slice<1,Shape::size()>(),stride().template slice<1,Shape::size()>(),off,data());
    }
    
    template <typename D = void> enable_if_one_dimensional<const T &,D> operator[](size_t i)const{
      auto off = offset() + i*stride().template get<0>();
#ifndef NDEBUG
      if(i>=shape().template get<0>()) throw std::range_error("invalid array index");
#endif
      return data()[off];
    }
    
    template <typename Idx> disable_if_one_dimensional<ElementType,Idx> operator[](Idx i){
      auto off = offset() + i*stride().template get<0>();
#ifndef NDEBUG
      if(i>=shape().template get<0>()) throw std::range_error("invalid array index");
#endif
      return ElementType(shape().template slice<1,Shape::size()>(),stride().template slice<1,Shape::size()>(),off,data());
    }
    
    template <typename D = void> enable_if_one_dimensional<T &,D> operator[](size_t i){
      auto off = offset() + i*stride().template get<0>();
#ifndef NDEBUG
      if(i>=shape().template get<0>()) throw std::range_error("invalid array index");
#endif
      return data()[off];
    }
    
    struct ContentSetter{
      NDArrayBase * parent;
      size_t i;
      ContentSetter(NDArrayBase *p):parent(p),i(0){}
      template <class Ty,class Sh, class St, class Of, class Da, class Cr> ContentSetter & operator,(const NDArrayBase<Ty,Sh,St,Of,Da,Cr> &other){
        for(auto j:range(other.size())){ (*parent)[i] = other[j]; ++i; }
        return *this;
      }
      ContentSetter & operator,(const T &x){ (*parent)[i] = x; ++i; return *this; }
      //    template <class E> ContentSetter & operator,(const E &x){ (*parent)[i] = x; ++i; return *this; }
    };
    
    template <class E> ContentSetter operator<<(const E &x){
      auto setter = ContentSetter(this);
      setter,x;
      return setter;
    }
    
    template <class C> void set_from(const C &c){ size_t i=0; for(auto v:c){ (*this)[i] = v; i++; } return *(Super*)this; }
    
    Super & operator=(const T &value){
      fill(value);
      return *(Super *)this;
    }
    
    Super & operator=(const std::initializer_list<Scalar> &c){ size_t i=0; for(auto v:c){ (*this)[i] = v; i++; } return *(Super*)this; }
    Super & operator=(const NDArrayBase &other){ return operator=<>(other); }
    
    template <typename oT,typename oShape,typename oStride,typename oOffset,typename oData,typename oCreator> typename std::enable_if<oShape::size() == Shape::size(),Super&>::type operator=(const NDArrayBase<oT, oShape, oStride, oOffset, oData, oCreator> & other){
#ifndef NDEBUG
      if(other.size() != size()) throw std::runtime_error("invalid assignment");
#endif
      for(auto i:range(size())) (*this)[i] = other[i];
      return *(Super*)this;
    }
    
    bool operator==(const NDArrayBase &other)const{
      if(other.size() != size()) return false;
      for(auto i:range(size())) if( !( (*this)[i] == other[i] ) ) return false;
      return true;
    }
    
    bool operator!=(const NDArrayBase &other)const{
      return !((*this) == other);
    }
    
    template <typename oT,typename oShape,typename oStride,typename oOffset,typename oData,typename oCreator>
    //typename std::enable_if<oShape::size() != Shape::size(),bool>::type
    bool operator==(const NDArrayBase<oT, oShape, oStride, oOffset, oData, oCreator> & other)const{
      if(other.size() != size()) return false;
      for(auto i:range(size())) if( !( (*this)[i] == other[i] ) ) return false;
      return true;
    }
    
    template <typename oT,typename oShape,typename oStride,typename oOffset,typename oData,typename oCreator> typename std::enable_if<oShape::size() == Shape::size(),bool>::type operator!=(const NDArrayBase<oT, oShape, oStride, oOffset, oData, oCreator> & other)const{
      return !((*this) == other);
    }
    
    template <class D=void> disable_if_one_dimensional<void, D> fill(const T &value){
      for(auto row:*this) row.fill(value);
    }
    
    template <class D=void> enable_if_one_dimensional<void, D> fill(const T &value){
      for(auto & v:*this) v = value;
    }
    
    template <typename F,typename Idx> enable_if_one_dimensional<void,Idx> for_all_indices_helper(F f,Idx idx)const{
      auto i = idx.push_back(0);
      for(auto j:range(size())){
        i.template set<Idx::size()>(j);
        f(i);
      }
    }
    
    template <typename F,typename Idx> disable_if_one_dimensional<void,Idx> for_all_indices_helper(F f,Idx idx)const{
      auto i = idx.push_back(0);
      for(auto j:range(size())){
        i.template set<Idx::size()>(j);
        (*this)[j].for_all_indices_helper(f,i);
      }
    }
    
    template <typename F> void for_all_indices(F f){
      for_all_indices_helper(f, IndexTuple<>());
    }
    
    template <typename F> void for_all_indices(F f)const{
      for_all_indices_helper(f, IndexTuple<>());
    }
    
    
    template <typename F> void element_wise(F f){
      for_all_indices( [&](Index idx){ (*this)(idx) = f(idx); } );
    }
    
    template <typename F,typename Idx> enable_if_one_dimensional<void,Idx> for_all_lower_indices_helper(F f,Idx idx,size_t max){
      auto i = idx.push_back(0);
      for(auto j:range(max)){
        i.template set<Idx::size()>(j);
        f(i);
      }
    }
    
    template <typename F,typename Idx> disable_if_one_dimensional<void,Idx> for_all_lower_indices_helper(F f,Idx idx,size_t max){
      auto i = idx.push_back(0);
      for(auto j:range(max)){
        i.template set<Idx::size()>(j);
        (*this)[j].for_all_lower_indices_helper(f,i,j);
      }
    }
    
    template <typename F> void for_all_lower_indices(F f){
      for_all_lower_indices_helper(f, IndexTuple<>(),size());
    }
    
    template <typename F,typename Idx> enable_if_one_dimensional<void,Idx> for_all_upper_indices_helper(F f,Idx idx,size_t min){
      auto i = idx.push_back(0);
      for(auto j:range(min,size())){
        i.template set<Idx::size()>(j);
        f(i);
      }
    }
    
    template <typename F,typename Idx> disable_if_one_dimensional<void,Idx> for_all_upper_indices_helper(F f,Idx idx,size_t min){
      auto i = idx.push_back(0);
      for(auto j:range(min,size())){
        i.template set<Idx::size()>(j);
        (*this)[j].for_all_upper_indices_helper(f,i,j+1);
      }
    }
    
    template <typename F> void for_all_upper_indices(F f){
      for_all_upper_indices_helper(f, IndexTuple<>(),0);
    }
    
    template <typename F,typename Idx> enable_if_one_dimensional<void,Idx> for_all_diagonal_indices_helper(F f,Idx idx,size_t j){
      auto i = idx.push_back(j);
      i.template set<Idx::size()>(j);
      f(i);
    }
    
    template <typename F,typename Idx> disable_if_one_dimensional<void,Idx> for_all_diagonal_indices_helper(F f,Idx idx,size_t j){
      auto i = idx.push_back(j);
      i.template set<Idx::size()>(j);
      (*this)[j].for_all_diagonal_indices_helper(f,i,j);
    }
    
    template <typename F> void for_all_diagonal_indices(F f){
      for(auto i:range(size()))for_all_diagonal_indices_helper(f, IndexTuple<>(),i);
    }
    
    template <typename F> void for_all_values(F f){
      for_all_indices([&](const Index &idx){ f((*this)(idx)); });
    }
    
    template <typename F> void for_all_values(F f)const{
      for_all_indices([&](const Index &idx){ f((*this)(idx)); });
    }
    
    using Transposed = NDArray<T, ReversedIndexTuple<Shape> , ReversedIndexTuple<Stride>, Offset, BorrowedData<T>>;
    using ConstTransposed = NDArray<const T, ReversedIndexTuple<Shape> , ReversedIndexTuple<Stride>, Offset, BorrowedData<const T>>;
    
    Transposed transpose(){
      return Transposed(reverse(shape()),reverse(stride()),offset(),data());
    }
    
    ConstTransposed transpose()const{
      return ConstTransposed(reverse(shape()),reverse(stride()),offset(),data());
    }
    
    Copy copy()const{
      Copy res(shape());
      res = *this;
      return res;
    }
    
    template <class F> Copy copy(F f)const{
      Copy res(shape());
      for_all_indices([&](Index i){ res(i) = f((*this)(i)); });
      return res;
    }
    
    
    // Operators
    
    
    template <class OtherShape,class Res> using enable_if_same_dim = typename std::enable_if<OtherShape::size() == Shape::size(),Res>::type;
    
    template <class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename Creator2>
    enable_if_same_dim<Shape2,CopyWithType<bool>> operator&&(const NDArrayBase<T2,Shape2,Stride2,Offset2,Data2,Creator2> &other)const{
#ifndef NDEBUG
      if(shape() != other.shape()) throw std::invalid_argument("comparison of arrays with different size");
#endif
      CopyWithType<bool> res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) && other(idx); });
      return res;
    }
    
    template <class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename Creator2>
    enable_if_same_dim<Shape2,CopyWithType<bool>> operator||(const NDArrayBase<T2,Shape2,Stride2,Offset2,Data2,Creator2> &other)const{
#ifndef NDEBUG
      if(shape() != other.shape()) throw std::invalid_argument("comparison of arrays with different size");
#endif
      CopyWithType<bool> res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) || other(idx); });
      return res;
    }
    
    template <class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename Creator2>
    enable_if_same_dim<Shape2,CopyWithType<bool>> operator>(const NDArrayBase<T2,Shape2,Stride2,Offset2,Data2,Creator2> &other)const{
#ifndef NDEBUG
      if(shape() != other.shape()) throw std::invalid_argument("comparison of arrays with different size");
#endif
      CopyWithType<bool> res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) > other(idx); });
      return res;
    }
    
    template <class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename Creator2>
    enable_if_same_dim<Shape2,CopyWithType<bool>> operator<(const NDArrayBase<T2,Shape2,Stride2,Offset2,Data2,Creator2> &other)const{
#ifndef NDEBUG
      if(shape() != other.shape()) throw std::invalid_argument("comparison of arrays with different size");
#endif
      CopyWithType<bool> res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) < other(idx); });
      return res;
    }
    
    template <class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename Creator2>
    enable_if_same_dim<Shape2,CopyWithType<bool>> operator>=(const NDArrayBase<T2,Shape2,Stride2,Offset2,Data2,Creator2> &other)const{
#ifndef NDEBUG
      if(shape() != other.shape()) throw std::invalid_argument("comparison of arrays with different size");
#endif
      CopyWithType<bool> res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) >= other(idx); });
      return res;
    }
    
    template <class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename Creator2>
    enable_if_same_dim<Shape2,CopyWithType<bool>> operator<=(const NDArrayBase<T2,Shape2,Stride2,Offset2,Data2,Creator2> &other)const{
#ifndef NDEBUG
      if(shape() != other.shape()) throw std::invalid_argument("comparison of arrays with different size");
#endif
      CopyWithType<bool> res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) <= other(idx); });
      return res;
    }
    
    template <class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename Creator2>
    enable_if_same_dim<Shape2,CopyWithType<bool>> element_wise_equal(const NDArrayBase<T2,Shape2,Stride2,Offset2,Data2,Creator2> &other)const{
#ifndef NDEBUG
      if(shape() != other.shape()) throw std::invalid_argument("comparison of arrays with different size");
#endif
      CopyWithType<bool> res(shape());
      res.for_all_indices([&](Index idx){ res(idx) == (*this)(idx) <= other(idx); });
      return res;
    }
    
    template <class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename Creator2>
    enable_if_same_dim<Shape2,Copy> operator+(const NDArrayBase<T2,Shape2,Stride2,Offset2,Data2,Creator2> &other)const{
#ifndef NDEBUG
      if(shape() != other.shape()) throw std::invalid_argument("summation of arrays with different size");
#endif
      Copy res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) + other(idx); });
      return res;
    }
    
    template <class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename Creator2>
    enable_if_same_dim<Shape2,Copy> operator-(const NDArrayBase<T2,Shape2,Stride2,Offset2,Data2,Creator2> &other)const{
#ifndef NDEBUG
      if(shape() != other.shape()) throw std::invalid_argument("subtraction of arrays with different size");
#endif
      Copy res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) - other(idx); });
      return res;
    }
    
    template <class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename Creator2>
    enable_if_same_dim<Shape2,Super> & operator+=(const NDArrayBase<T2,Shape2,Stride2,Offset2,Data2,Creator2> &other){
#ifndef NDEBUG
      if(shape() != other.shape()) throw std::invalid_argument("summation of arrays with different size");
#endif
      this->for_all_indices([&](Index idx){ (*this)(idx) += other(idx); });
      return *(Super*)this;
    }
    
    template <class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename Creator2>
    enable_if_same_dim<Shape2,Super> & operator-=(const NDArrayBase<T2,Shape2,Stride2,Offset2,Data2,Creator2> &other){
#ifndef NDEBUG
      if(shape() != other.shape()) throw std::invalid_argument("subtraction of arrays with different size");
#endif
      this->for_all_indices([&](Index idx){ (*this)(idx) -= other(idx); });
      return *(Super*)this;
    }
    
    template <class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename Creator2>
    enable_if_same_dim<Shape2,Copy> operator*(const NDArrayBase<T2,Shape2,Stride2,Offset2,Data2,Creator2> &other)const{
#ifndef NDEBUG
      if(shape() != other.shape()) throw std::invalid_argument("product of arrays with different size");
#endif
      Copy res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) * other(idx); });
      return res;
    }
    
    template <class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename Creator2>
    enable_if_same_dim<Shape2,Copy> operator/(const NDArrayBase<T2,Shape2,Stride2,Offset2,Data2,Creator2> &other)const{
#ifndef NDEBUG
      if(shape() != other.shape()) throw std::invalid_argument("summation of arrays with different size");
#endif
      Copy res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) / other(idx); });
      return res;
    }
    
    template <class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename Creator2>
    enable_if_same_dim<Shape2,Super> & operator*=(const NDArrayBase<T2,Shape2,Stride2,Offset2,Data2,Creator2> &other){
#ifndef NDEBUG
      if(shape() != other.shape()) throw std::invalid_argument("product of arrays with different size");
#endif
      this->for_all_indices([&](Index idx){ (*this)(idx) *= other(idx); });
      return *(Super*)this;
    }
    
    template <class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename Creator2>
    enable_if_same_dim<Shape2,Super> & operator/=(const NDArrayBase<T2,Shape2,Stride2,Offset2,Data2,Creator2> &other){
#ifndef NDEBUG
      if(shape() != other.shape()) throw std::invalid_argument("quotient of arrays with different size");
#endif
      this->for_all_indices([&](Index idx){ (*this)(idx) /= other(idx); });
      return *(Super*)this;
    }
    
    CopyWithType<bool> operator&&(const Scalar &s)const{
      CopyWithType<bool> res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) && s; });
      return res;
    }
    
    CopyWithType<bool> operator||(const Scalar &s)const{
      CopyWithType<bool> res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) || s; });
      return res;
    }
    
    CopyWithType<bool> operator>(const Scalar &s)const{
      CopyWithType<bool> res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) > s; });
      return res;
    }
    
    CopyWithType<bool> operator<(const Scalar &s)const{
      CopyWithType<bool> res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) < s; });
      return res;
    }
    
    CopyWithType<bool> operator>=(const Scalar &s)const{
      CopyWithType<bool> res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) >= s; });
      return res;
    }
    
    CopyWithType<bool> operator<=(const Scalar &s)const{
      CopyWithType<bool> res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) <= s; });
      return res;
    }
    
    CopyWithType<bool> operator==(const Scalar &s)const{
      CopyWithType<bool> res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) == s; });
      return res;
    }
    
    CopyWithType<bool> operator!=(const Scalar &s)const{
      CopyWithType<bool> res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) != s; });
      return res;
    }
    
    Copy operator*(const Scalar &s)const{
      Copy res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) * s; });
      return res;
    }
    
    Copy operator/(const Scalar &s)const{
      Copy res(shape());
      res.for_all_indices([&](Index idx){ res(idx) = (*this)(idx) / s; });
      return res;
    }
    
    Super & operator*=(const Scalar &s){
      this->for_all_indices([&](Index idx){ (*this)(idx) *= s; });
      return *(Super *)this;
    }
    
    Super & operator/=(const Scalar &s){
      this->for_all_indices([&](Index idx){ (*this)(idx) /= s; });
      return *(Super *)this;
    }
    
    Copy operator-()const{
      return (*this) * -1;
    }
    
    Scalar sum()const{
      Scalar res;
      res = 0;
      for_all_values([&](const Scalar &v){ res += v; });
      return res;
    }

    Scalar front()const{
        return (*this)(IndexTupleRepeat<0,ndim()>());
    }

    Scalar back()const{
        return (*this)(shape() - IndexTupleRepeat<1,ndim()>());
    }
    
    Scalar &front(){
        return (*this)(IndexTupleRepeat<0,ndim()>());
    }

    Scalar &back(){
        return (*this)(shape() - IndexTupleRepeat<1,ndim()>());
    }
    
    Scalar max()const{
      Scalar res;
      res = front();
      for_all_values([&](const Scalar &v){ res = v>res?v:res; });
      return res;
    }
    
    Scalar min()const{
      Scalar res;
      res = front();
      for_all_values([&](const Scalar &v){ res = v>res?res:v; });
      return res;
    }
    
    size_t count()const{ return NDArrayCalculator<Shape>::prod(shape()).template get<0>(); }
    Scalar average()const{ return sum()/count(); }
    
    Scalar norm_squared()const{
      Scalar res = 0;
      this->for_all_indices([&](Index idx){ res += (*this)(idx) * (*this)(idx); });
      return res;
    }
    
    Scalar norm()const{
      using namespace std;
      return sqrt(norm_squared());
    }
    
    Copy normalized()const{ return (*this)/norm(); }
    
    NDArrayBase &as_array(){ return *this; }
    const NDArrayBase &as_array()const{ return *this; }
  };
  
  // Commutative operators
  
  template <class T,typename Shape,typename Stride,typename Offset,typename Data,typename Creator>
  typename NDArrayBase<T,Shape,Stride,Offset,Data,Creator>::Copy operator*(const typename NDArrayBase<T,Shape,Stride,Offset,Data,Creator>::Scalar &s, const NDArrayBase<T,Shape,Stride,Offset,Data,Creator> &m ){
    return m*s;
  }
  
  template <class T,typename Shape,typename Stride,typename Offset,typename Data,typename Creator,
  class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename Creator2>
  typename NDArrayBase<T,Shape,Stride,Offset,Data,Creator>::Copy operator/(const typename NDArrayBase<T,Shape,Stride,Offset,Data,Creator>::Scalar &s, const NDArrayBase<T2,Shape2,Stride2,Offset2,Data2,Creator2> &m ){
    return m.copy([&](const T &v){ return s/v; });
  }
  
  // Functions
  template <class T,typename Shape,typename Stride,typename Offset,typename Data,typename Creator>
  typename NDArrayBase<T,Shape,Stride,Offset,Data,Creator>::Copy
  abs(const NDArrayBase<T,Shape,Stride,Offset,Data,Creator> &arr){
    return arr.copy([](const T &v){ return v>0?v:-v; });
  }
  
  template <class T,typename Shape,typename Stride,typename Offset,typename Data,typename Creator>
  typename NDArrayBase<T,Shape,Stride,Offset,Data,Creator>::Copy
  sqrt(const NDArrayBase<T,Shape,Stride,Offset,Data,Creator> &arr){
    return arr.copy([](const T &v){ return std::sqrt(v); });
  }
  
  template <class T,typename Shape,typename Stride,typename Offset,typename Data,typename Creator,class E>
  typename NDArrayBase<T,Shape,Stride,Offset,Data,Creator>::Copy
  pow(const NDArrayBase<T,Shape,Stride,Offset,Data,Creator> &arr,const E &e){
    return arr.copy([&](const T &v){ return std::pow(v,e); });
  }
  
  // Basic Creator
  
  template <template <class,class,class> class NewType> struct BasicNDArrayCreator{
    template <class T,typename Shape,typename Stride,typename Offset,typename Data> using NDArray = NDArrayBase<T, Shape, Stride, Offset, Data, BasicNDArrayCreator>;
    template<class T,class Shape> using NewNDArray = NewType<T,Shape,BasicNDArrayCreator>;
  };
  
  template <typename Char, typename Traits,class T,typename Shape,typename Stride,typename Offset,typename Data, typename Creator>
  std::basic_ostream<Char, Traits> & operator<<(std::basic_ostream<Char, Traits> &stream,const NDArrayBase<T, Shape, Stride, Offset, Data, Creator> & array){
    
    if(array.ndim() == 2 && array.shape().template safe_get<1>() == 1 && array.shape().template safe_get<0>() > 1){
      stream << array.transpose() << "^t";
      return stream;
    }
    
    stream << '[';
    if(array.size() != 0) for(auto i:range(array.size()-1)){
      stream << array[i] << ',';
      for(auto i UNUSED:range(array.ndim()-1)) stream << '\n';
    }
    if(array.size() != 0){
      stream << array[array.size()-1];
    }
    stream << ']';
    return stream;
  }
  
  template <class T> class HeapData{
    T * data = nullptr;
    size_t size = 0;
    
  public:
    
    HeapData(size_t size){ resize(size); }
    
    HeapData(const HeapData &other){ resize(other.size); for(auto i:range(size)) data[i] = other.data[i]; }
    HeapData(HeapData && other){ std::swap(data,other.data); std::swap(size,other.size); }
    HeapData &operator=(HeapData && other){ std::swap(data,other.data); std::swap(size,other.size);  return *this; }
    HeapData &operator=(const HeapData &other){ resize(other.size); for(auto i:range(size)) data[i] = other.data[i]; return *this; }
    
    const T * get()const{ return data; }
    T * get(){ return data; }
    void resize(size_t _size){ size = _size; if(data) delete[]data; data = new T[size]; }
    
    ~HeapData(){ if(data) delete[]data; }
  };
  
  template <class T,size_t size> class StackData{
    std::array<T,size> data;
  public:
    StackData(const std::array<T,size> &arr):data(arr){}
    
    StackData() = default;
    StackData(const StackData &) = default;
    StackData(StackData &&) = default;
    StackData &operator=(const StackData &) = default;
    StackData &operator=(StackData &&) = default;
    
    T * get(){ return &data[0]; }
    const T * get()const{ return &data[0]; }
  };
  
  template <class T,typename Shape,class Creator> class HeapNDArray;
  
  template <class T,typename Shape,class Creator = BasicNDArrayCreator<HeapNDArray>> class HeapNDArray:public Creator::template NDArray<T,Shape, typename NDArrayCalculator<Shape>::Stride, StaticIndex<0>,HeapData<T>>{
  public:
    
    using Base = typename Creator::template NDArray<T,Shape, typename NDArrayCalculator<Shape>::Stride, StaticIndex<0>,HeapData<T>>;
    using Base::operator=;
    
    HeapData<T> & get_data(){ return Base::_data; }
    const HeapData<T> & get_data()const{ return Base::_data; }
    
    HeapNDArray(Shape shape = Shape()):Base(shape,NDArrayCalculator<Shape>::stride(shape),StaticIndex<0>(),NDArrayCalculator<Shape>::prod(shape).template get<0>()){}
    
    template <typename Shape2> HeapNDArray(HeapNDArray<T,Shape2,Creator> && other):Base(other.shape(),NDArrayCalculator<Shape>::stride(other.shape()),StaticIndex<0>(),std::move(other.get_data())){ }
    
    // template <typename ... Size> HeapNDArray(Size ... size):Base(make_dynamic_index_tuple(size...) ,NDArrayCalculator<Shape>::stride(make_dynamic_index_tuple(size...) ),StaticIndex<0>(),NDArrayCalculator<Shape>::prod(make_dynamic_index_tuple(size...) ).template get<0>()){}
    
    
    template <typename Shape2> HeapNDArray & operator=(HeapNDArray<T,Shape2,Creator> && other){
      Base::_data = std::move(other.get_data());
      Base::SShape::operator=(other);
      Base::SStride::operator=(other);
      Base::SOffset::operator=(other);
      return *this;
    }
    
    HeapNDArray(HeapNDArray && other):Base(other.shape(),NDArrayCalculator<Shape>::stride(other.shape()),StaticIndex<0>(),std::move(other.get_data())){ }
    
    HeapNDArray & operator=(HeapNDArray && other){
      Base::swap(other);
      return *this;
    }
    
    HeapNDArray(const HeapNDArray& other):Base(other.shape(),NDArrayCalculator<Shape>::stride(other.shape()),StaticIndex<0>(),other.get_data()){
      
    }
    
    HeapNDArray & operator=(const HeapNDArray& other){
      resize(other.shape());
      Base::_data = other.get_data();
      return *this;
    }
    
    template <typename Shape2> HeapNDArray(const HeapNDArray<T,Shape2,Creator> & other):Base(other.shape(),NDArrayCalculator<Shape>::stride(other.shape()),StaticIndex<0>(),other.get_data()){ }
    
    template <typename oT,typename oShape,typename oStride,typename oOffset,typename oData,class oCreator>
    
    HeapNDArray(const NDArrayBase<oT, oShape, oStride, oOffset, oData, oCreator> & other):Base(other.shape(),NDArrayCalculator<Shape>::stride(other.shape()),StaticIndex<0>(),NDArrayCalculator<Shape>::prod(other.shape()).template get<0>()){
      Base::operator=(other);
    }
    
    template <typename oT,typename oShape,typename oStride,typename oOffset,typename oData,typename oCreator>
    typename std::enable_if<oShape::size() == Shape::size(),HeapNDArray&>::type
    operator=(const NDArrayBase<oT, oShape, oStride, oOffset, oData, oCreator> & other){
      resize(other.shape());
      Base::operator=(other);
      return *this;
    }
    
    template <typename S> void resize(S shape){
      Base::SShape::set(shape);
      Base::SStride::set(NDArrayCalculator<Shape>::stride(shape));
      Base::SOffset::set(StaticIndex<0>());
      Base::_data.resize(NDArrayCalculator<Shape>::prod(shape).template get<0>());
    }
    
    template <typename ... Args> void resize(size_t first,Args ... rest){ resize(Shape(first,rest...)); }
    
    void transpose_in_place(){
      Base::for_all_lower_indices([&](typename Base::Index idx){ std::swap((*this)(idx),(*this)(reverse(idx))); });
    }
    
  };
  
  template <class T,typename Shape,class Creator> class StackNDArray;
  
  template <class T,typename Shape,class Creator = BasicNDArrayCreator<StackNDArray>> class StackNDArray:public Creator::template
  NDArray<T,Shape, typename NDArrayCalculator<Shape>::Stride, StaticIndex<0>,StackData<T, NDArrayCalculator<Shape>::Prod::template get<0>() > >{
  public:
    
    using Base = typename Creator::template NDArray<T,Shape, typename NDArrayCalculator<Shape>::Stride, StaticIndex<0>,StackData<T, NDArrayCalculator<Shape>::Prod::template get<0>() > >;
    
    using Base::operator=;
    using Base::Base;
    
    template <typename ... Args, class E = typename std::enable_if<sizeof...(Args) + 1 == NDArrayCalculator<Shape>::Prod::template get<0>()>::type>
    StackNDArray(const T &first, Args ... args):Base(Shape(),NDArrayCalculator<Shape>::stride(Shape()),StaticIndex<0>(),std::array<T,NDArrayCalculator<Shape>::Prod::template get<0>()>{{first,static_cast<T>(args)...}}){
      // static_assert(sizeof...(args) + 1 == NDArrayCalculator<Shape>::Prod::template get<0>(),"initialization arguments must match array size");
    }
    
    StackNDArray(Shape shape = Shape()):Base(shape,NDArrayCalculator<Shape>::stride(shape),StaticIndex<0>()){}
    
    StackNDArray(const StackNDArray &other):Base(other.shape(),other.stride(),StaticIndex<0>()){
      Base::operator=(other);
    }
    
    template <typename S> void resize(S shape){ Base::SShape::set(shape); }
    template <typename ... Args> void resize(size_t first,Args ... rest){ resize(Shape(first,rest...)); }
    
  };
  
  //template <class T,size_t D, template<class,class> class Array = HeapNDArray> using NDArray = Array<T, DynamicIndexTuple<D>>;
  //template <class T,class Shape,class Stride, class Offset,class Creator>
  //using MappedNDArray = NDArrayBase<T, Shape, Stride, Offset, BorrowedData<T>, Creator>;
  
  template <class T, template<class,class,class> class Array = HeapNDArray,class C = BasicNDArrayCreator<HeapNDArray>,typename Shape> Array<T, Shape,C> make_ndarray(Shape shape){
    return Array<T, Shape, C>(shape);
  }
  
  
}



