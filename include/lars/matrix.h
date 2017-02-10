#pragma once

#include <lars/ndarray.h>

namespace lars {
  
  struct SingularMatrixException:public std::exception{
    const char * what()const noexcept{ return "singular matrix encountered"; }
  };
  
  namespace matrix_algorithms{
    
    // From numerical recipies in c second edition
    template <class A,class P> int LUP_decompose_inplace(A & a, P & p){
      
      using std::abs;
      
      size_t i,imax = 0,j,k,n = a.size();
      int d=1;
      typename A::Scalar big,dum,sum,temp;
      
      using VectorSize = typename A::Shape::template Slice<0,1>;
      typename A::Creator::template NewNDArray<typename A::Scalar,VectorSize> vv(a.shape().template slice<0,1>());
      
      for (i=0;i<n;i++) {
        big=0.0;
        for (j=0;j<n;j++) if ((temp = abs(a[i][j])) > big) big=temp;
        if (big == 0.0) throw SingularMatrixException();
        vv[i]=1.0/big;
      }
      
      for (j=0;j<n;j++) {
        for (i=0;i<j;i++) {
          sum = a[i][j];
          for (k=0;k<i;k++) sum -= a[i][k]*a[k][j];
          a[i][j]=sum;
        }
        big=0.0;
        for (i=j;i<n;i++) {
          sum=a[i][j];
          for (k=0;k<j;k++)
            sum -= a[i][k]*a[k][j];
          a[i][j]=sum;
          if ( (dum = vv[i] * abs(sum)) >= big) {
            big=dum;
            imax=i;
          }
        }
        if (j != imax) {
          for (k=0;k<n;k++) {
            dum=a[imax][k];
            a[imax][k]=a[j][k];
            a[j][k]=dum;
          }
          d = -d;
          vv[imax]=vv[j];
        }
        p[j]=imax;
        if (a[j][j] == 0.0) throw SingularMatrixException(); // originally: a[j-1][j-1]=1.0e-20, but yields wrong results if matrix is actually singular;
        if (j != n) {
          dum=1.0/(a[j][j]);
          for (i=j+1;i<n;i++) a[i][j] *= dum;
        }
      }
      
      return d;
    }
    
    template <class M> struct LUP_Decomposition{
      using IndexArray = typename M::Creator::template NewNDArray<size_t,typename M::Shape::template Slice<0,1>>;
      using LUMatrix = typename M::Creator::template NewNDArray<typename M::Scalar,typename M::Shape>;
      
      IndexArray P;
      LUMatrix LU;
      int d;
      
      LUP_Decomposition(const M &mat){ LU.resize(mat.shape()); P.resize(mat.size()); LU = mat; d = LUP_decompose_inplace(LU, P); }
      LUP_Decomposition(M && mat):LU(mat){ d = LUP_decompose_inplace(LU, P);  }
    };
    
    template <class M> LUP_Decomposition<M> LUP_decompose(const M &mat){
      return LUP_Decomposition<M>(mat);
    }
    
    template <class M> LUP_Decomposition<M> LUP_decompose(M && mat){
      return LUP_Decomposition<M>(mat);
    }
    
    template <class LU,class P,class B> void LUP_solve_inplace(const LU &a, const P &indx, B & b){
      auto n = a.size();
      size_t i,ii=0,ip,j;
      typename LU::Scalar sum;
      for (i=0;i<n;i++) {
        ip=indx[i];
        sum=b[ip];
        b[ip]=b[i];
        if(ii+1) for (j=ii;j<i;j++) sum -= a[i][j]*b[j];
        else if(sum) ii=i;
        b[i]=sum;
      }
      for (i=n-1;i<n;i--) {
        sum=b[i];
        for (j=i+1;j<n;j++) sum -= a[i][j]*b[j];
        b[i]=sum/a[i][i];
      }
    }
    
    template <class M,class V> typename V::Copy LUP_solve(const LUP_Decomposition<M> &LUP,const V &b){
      typename V::Copy x = b.copy();
      LUP_solve_inplace(LUP.LU, LUP.P, x);
      return x;
    }
    
    template <class M,class V> void LUP_solve_inplace(const LUP_Decomposition<M> &LUP, V &b){
      LUP_solve_inplace(LUP.LU, LUP.P, b);
    }
    
    template <class M> typename M::Copy LUP_inverse(const LUP_Decomposition<M> &LUP){
      typename M::Copy inv(LUP.LU.shape());
      inv.fill(0);
      for(auto j=0;j<LUP.LU.size();j++) {
        auto col = inv.transpose()[j];
        col[j]=1.0;
        LUP_solve_inplace(LUP,col);
      }
      return inv;
    }
    
    template <class M> typename M::Copy LUP_inverse(const M &mat){
      return LUP_inverse(LUP_decompose(mat));
    }
    
    template <class M> typename M::Scalar determinant_from_LU(const LUP_Decomposition<M> &LUP){
      typename M::Scalar det = LUP.d;
      for(size_t i=0;i<LUP.LU.size();i++) det *= LUP.LU[i][i];
      return det;
    }
    
    template <class M> typename M::Scalar explicit_1D_determinant(const M &m){
      return m[0][0];
    }
    
    template <class M> typename M::Scalar explicit_2D_determinant(const M &m){
      return m[0][0] * m[1][1] - m[0][1] * m[1][0];
    }

    template <class M> typename M::Scalar explicit_3D_determinant(const M &m){
      return m[0][0] * ( m[1][1] * m[2][2] - m[1][2] * m[2][1] )
           - m[0][1] * ( m[1][0] * m[2][2] - m[1][2] * m[2][0] )
           + m[0][2] * ( m[1][0] * m[2][1] - m[1][1] * m[2][0] );
    }

    template <class M> typename M::Copy explicit_1D_inverse(const M &m){
      auto det = explicit_1D_determinant(m);
      if(det == 0) throw SingularMatrixException();
      typename M::Copy inv(m.shape());
      inv[0][0] = 1/det;
      return inv;
    }
    
    template <class M> typename M::Copy explicit_2D_inverse(const M &m){
      auto det = explicit_2D_determinant(m);
      if(det == 0) throw SingularMatrixException();
      auto inv = typename M::Copy(m.shape());
      inv[0] <<  m[1][1] , -m[0][1];
      inv[1] << -m[1][0] ,  m[0][0];
      inv.for_all_indices([&](typename M::Copy::Index idx){ inv(idx) /= det; });
      return inv;
    }
    
    template <class M> typename M::Copy explicit_3D_inverse(const M &m){
      auto det = explicit_3D_determinant(m);
      if(det == 0) throw SingularMatrixException();
      auto inv = typename M::Copy(m.shape());
      inv[0] << m[1][1] * m[2][2] - m[1][2] * m[2][1], m[0][2] * m[2][1] - m[0][1] * m[2][2], m[0][1] * m[1][2] - m[0][2] * m[1][1];
      inv[1] << m[1][2] * m[2][0] - m[1][0] * m[2][2], m[0][0] * m[2][2] - m[0][2] * m[2][0], m[0][2] * m[1][0] - m[0][0] * m[1][2];
      inv[2] << m[1][0] * m[2][1] - m[1][1] * m[2][0], m[0][1] * m[2][0] - m[0][0] * m[2][1], m[0][0] * m[1][1] - m[0][1] * m[1][0];
      inv.for_all_indices([&](typename M::Copy::Index idx){ inv(idx) /= det; });
      return inv;
    }
    
  }
  
  template <class T,typename Shape,typename Stride,typename Offset,typename Data,typename MatrixCreator> class MatrixBase:public NDArrayBase<T, Shape, Stride, Offset, Data, MatrixCreator>{
    
  public:
    
    using Base = NDArrayBase<T, Shape, Stride, Offset, Data, MatrixCreator>;

    using Base::Base;
    using Base::operator=;
    using Base::ndim;
    using Base::shape;
    using Scalar = typename Base::Scalar;
    using Index = typename Base::Index;
    using Copy = typename Base::Copy;
    using Super = typename Base::Super;
    
    using SingularMatrixException = lars::SingularMatrixException;
    
    static Copy create_identity(Shape s = Shape()){
      Copy m(s);
      m.for_all_indices([&](typename Copy::Index idx){
        m(idx) = idx.template get<0>() ==  idx.template get<1>();
      });
      return m;
    }
    
    template <typename OtherShape> using ProductShape = typename Shape::template Slice<0,1>::template Append<typename OtherShape::template Slice<1, 2>>;
    
    template <size_t n, class Res,class Dummy> using enable_if_n_dimensional_array = typename std::enable_if<Base::ndim() == n, typename std::conditional<true, Res, Dummy>::type >::type;
    template <size_t n, class Res,class Dummy> using enable_if_at_least_n_dimensional_array = typename std::enable_if<Base::ndim() >= n, typename std::conditional<true, Res, Dummy>::type >::type;
    
    size_t m()const{ return this->shape().template get<0>(); }
    template <class Dummy = void> enable_if_at_least_n_dimensional_array<2,size_t,Dummy> n()const{ return this->shape().template get<1>(); }
    
    static constexpr bool is_vector(){ return Shape::template safe_static_get<0>() == 1 || Shape::template safe_static_get<1>() == 1; }

    template <class Dummy = Scalar> typename std::enable_if<is_vector(),Dummy>::type & operator()(size_t i){
      if(m() == 1) return (*this)[0][i];
      else return (*this)[i][0];
    }
    
    template <class Dummy = Scalar> const typename std::enable_if<is_vector(),Dummy>::type & operator()(size_t i)const{
      if(m() == 1) return (*this)[0][i];
      else return (*this)[i][0];
    }
    
    template <typename ... Args> static const typename std::enable_if<is_vector(),typename MatrixCreator::template NewNDArray<Scalar, StaticIndexTuple<sizeof...(Args),1> > >::type create(Args ... args){
        typename MatrixCreator::template NewNDArray<Scalar, StaticIndexTuple<sizeof...(Args),1> > vec;
        vec = {static_cast<Scalar>(args)...};
        return vec;
    }
    
    using Base::operator();

    template <class Dummy = Scalar> typename std::enable_if<is_vector(),Dummy>::type & x(){ return (*this)(0); }
    template <class Dummy = Scalar> const typename std::enable_if<is_vector(),Dummy>::type & x()const{ return (*this)(0); }
    template <class Dummy = Scalar> typename std::enable_if<is_vector(),Dummy>::type & y(){ return (*this)(1); }
    template <class Dummy = Scalar> const typename std::enable_if<is_vector(),Dummy>::type & y()const{ return (*this)(1); }
    template <class Dummy = Scalar> typename std::enable_if<is_vector(),Dummy>::type & z(){ return (*this)(2); }
    template <class Dummy = Scalar> const typename std::enable_if<is_vector(),Dummy>::type & z()const{ return (*this)(2); }
    
    template <typename Other> using Product = typename MatrixCreator::template NewNDArray< typename std::remove_const<T>::type , ProductShape<typename Other::Shape>>;
    
    using Base::operator*;

    template <class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename MatrixCreator2>
    Product<MatrixBase<T2,Shape2,Stride2,Offset2,Data2,MatrixCreator2>>
    operator*(const MatrixBase<T2,Shape2,Stride2,Offset2,Data2,MatrixCreator2> &other)const{
#ifndef NDEBUG
      if(n() != other.m()) throw std::invalid_argument("incompatible matrices");
#endif
      using Prod = Product< MatrixBase<T2,Shape2,Stride2,Offset2,Data2,MatrixCreator2> >;
      Prod prod(typename Prod::Shape(m(),other.n()));
      
      prod = 0;
      prod.for_all_indices([&](typename Prod::Index idx){
        auto i = idx.template get<0>(); auto j = idx.template get<1>();
        for(auto k : range(other.m())) prod(idx) += (*this)(Index(i,k)) * other(Index(k,j));
      });
      return prod;
    }
    
    template <class T2,typename Shape2,typename Stride2,typename Offset2,typename Data2,typename MatrixCreator2>
    Scalar dot(const MatrixBase<T2,Shape2,Stride2,Offset2,Data2,MatrixCreator2> &other)const{
      auto res = (*this).transpose() * other;
#ifndef NDEBUG
      if(res.n() != 1 && res.m() != 1) throw std::invalid_argument("result of scalar product is not scalar");
#endif
      return res[0][0];
    }
 
    using LUP_Decomposition = matrix_algorithms::LUP_Decomposition<Super>;
    
    LUP_Decomposition LUP_decompose()const{ return LUP_Decomposition(*this); }
    
    template <class Dummy> using ShapeIf2D = enable_if_n_dimensional_array<2, Shape, Dummy>;
    template <size_t N,class Res,class Dummy> using enable_if_ND_matrix = typename std::enable_if< ShapeIf2D<Dummy>::template get<0>() == N && ShapeIf2D<Dummy>::template get<1>() == N, Res>::type;
    template <size_t N,class Res,class Dummy> using enable_if_at_least_ND_square_matrix = typename std::enable_if< ShapeIf2D<Dummy>::template get<0>() >= N && ShapeIf2D<Dummy>::template get<0>() == ShapeIf2D<Dummy>::template get<1>() , Res>::type;
    template <size_t N,class Res,class Dummy> using enable_if_dynamic_matrix = typename std::enable_if< ShapeIf2D<Dummy>::is_dynamic() , Res>::type;
    
    Scalar determinant()const{
#ifndef NDEBUG
      if(m() != n()) throw std::invalid_argument("calculating determinant of non-square matrix");
#endif
      if(m() == 1) return matrix_algorithms::explicit_1D_determinant(*this);
      else if(m() == 2) return matrix_algorithms::explicit_2D_determinant(*this);
      else if(m() == 3) return matrix_algorithms::explicit_3D_determinant(*this);
      try {
        return matrix_algorithms::determinant_from_LU(LUP_decompose());
      } catch (SingularMatrixException) {
        return 0;
      }
    }
    
    Copy inverse()const{
#ifndef NDEBUG
      if(m() != n()) throw std::invalid_argument("calculating inverse of non-square matrix");
#endif
      if(m() == 1) return matrix_algorithms::explicit_1D_inverse(*this);
      else if(m() == 2) return matrix_algorithms::explicit_2D_inverse(*this);
      else if(m() == 3) return matrix_algorithms::explicit_3D_inverse(*this);
      return matrix_algorithms::LUP_inverse(*this);
    }
    
    template <class R = std::enable_if<is_vector() && Shape::template safe_static_get<0>()==2> > Scalar angle(){
      return atan2(y(),x());
    }
    
    using Array = typename MatrixCreator::template MappedBasicNDArray<T,Shape,Stride,Offset>;
    
    Array as_array(){
      return Array(this->shape(),this->stride(),this->offset(), this->data());
    }
    
    using ConstArray = typename MatrixCreator::template MappedBasicNDArray<const T,Shape,Stride,Offset>;
    
    ConstArray as_array()const{
      return ConstArray(this->shape(),this->stride(),this->offset(), this->data());
    }
    
  };
      
  // Matrix Type
  template <template <class,class,class> class Array> struct MatrixCreator{
    template <class T,typename Shape,typename Stride,typename Offset,typename Data> using NDArray = MatrixBase<T, Shape, Stride, Offset, Data, MatrixCreator>;
    template <class T,typename Shape> using NewNDArray = Array<T, Shape , MatrixCreator> ;
    template <class T,typename Shape,typename Stride,typename Offset> using MappedBasicNDArray = NDArrayBase<T, Shape, Stride , Offset, BorrowedData<T>, BasicNDArrayCreator<Array>>;
  };
  
  template <class T,size_t m,size_t n> using Matrix = MatrixCreator<StackNDArray>::NewNDArray<T, StaticIndexTuple<m,n>>;;
  template <class T> using DynamicMatrix = MatrixCreator<HeapNDArray>::NewNDArray<T, DynamicIndexTuple<2>>;
    
}




