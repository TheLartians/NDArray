
#include <iostream>
#include <lars/matrix.h>
#include <lars/timeit.h>

using namespace lars;

template <size_t N> using matrixh = MatrixCreator<HeapNDArray>::NewNDArray<float, StaticIndexTuple<N,N>> ;
template <size_t N> using vectorh = MatrixCreator<HeapNDArray>::NewNDArray<float, StaticIndexTuple<N,1>>;
template <size_t N> using matrixs = MatrixCreator<StackNDArray>::NewNDArray<float, StaticIndexTuple<N,N>> ;
template <size_t N> using vectors = MatrixCreator<StackNDArray>::NewNDArray<float, StaticIndexTuple<N,1>>;

template <class M> typename M::Scalar __attribute__ ((noinline)) determinant(const M &m){
  return m.determinant();
}

template <class M,class V> typename V::Copy __attribute__ ((noinline)) multiplication(const M &m, const V & v){
  return m * m * v;
}

template <class M> typename M::Copy __attribute__ ((noinline)) inversion(const M &m){
  return m.inverse();
}

template <class M,class V> void measure(std::string type){
  M mat;
  V vec;
  
  mat.for_all_indices([&](typename decltype(mat)::Index idx){
    int i = idx.template get<0>(), j = idx.template get<1>(); mat(idx) = 2+i/(j+1);
  });
  
  vec.for_all_indices([&](typename decltype(mat)::Index idx){
    int i = idx.template get<0>(); vec(idx) = i + 1;
  });
  
  timeit<10000>(type + " multiplication " , [&](){ return multiplication(mat,vec); }  );
  timeit<10000>(type + " determinant " , [&](){ return determinant(mat); }  );
  timeit<10000>(type + " inversion " , [&](){ return inversion(mat); }  );
}

int main(){
  
  std::cout << "size = 2" << std::endl;
  measure<matrixs<2>,vectors<2>>("stack");
  measure<matrixh<2>,vectorh<2>>("heap");
  
  std::cout << std::endl;
  
  std::cout << "size = 3" << std::endl;
  measure<matrixs<3>,vectors<3>>("stack");
  measure<matrixh<3>,vectorh<3>>("heap");
  
  std::cout << std::endl;
  std::cout << "size = 4" << std::endl;
  measure<matrixs<4>,vectors<4>>("stack");
  measure<matrixh<4>,vectorh<4>>("heap");
  
  std::cout << std::endl;
  
  std::cout << "size = 24" << std::endl;
  measure<matrixs<24>,vectors<24>>("stack");
  measure<matrixh<24>,vectorh<24>>("heap");

  std::cout << std::endl;

  std::cout << "size = 128" << std::endl;
  measure<matrixs<128>,vectors<128>>("stack");
  measure<matrixh<128>,vectorh<128>>("heap");

  return 0;
}

