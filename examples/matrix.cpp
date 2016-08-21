
#include <iostream>
#include <lars/matrix.h>
#include <lars/timeit.h>

using namespace lars;

template <size_t size> void static_matrix_test(){
  
  std::cout << std::endl << "---------------------------------" << std::endl << "Static test: size = " << size << std::endl << "---------------------------------" << std::endl << std::endl;
  
  Matrix<double,size,size> mat;
  mat.for_all_indices([&](typename decltype(mat)::Index idx){
    int i = idx.template get<0>(), j = idx.template get<1>();
    mat(idx) = 2+i/(j+1);
  });
  
  Matrix<double,size,1> vec;
  vec.for_all_indices([&](typename decltype(mat)::Index idx){ int i = idx.template get<0>(); vec(idx) = i + 1; });
  
  std::cout << "mat = \n" << mat << std::endl << std::endl;
  
  std::cout << "mat[0] = " << mat[0] << std::endl << std::endl;
  std::cout << "mat.transpose()[0] = " << mat.transpose()[0] << std::endl << std::endl;
  
  std::cout << "vec = \n" << vec << std::endl << std::endl;
  std::cout << "vec.transpose() = " << vec.transpose() << std::endl << std::endl;
  
  std::cout << "mat * vec = \n" << mat * vec << std::endl << std::endl;
  std::cout << "mat.transpose() * mat = \n" << mat.transpose() * mat << std::endl << std::endl;
  std::cout << "vec.transpose() * mat = " << vec.transpose() * mat << std::endl << std::endl;
  std::cout << "vec * vec.transpose() = \n" << vec * vec.transpose() << std::endl << std::endl;
  
  std::cout << "vec.transpose() * vec = " << vec.transpose() * vec << std::endl << std::endl;
  
  float scalar = (vec.transpose() * vec)[0][0];
  std::cout << "float(vec.transpose() * vec) = " << scalar << std::endl << std::endl;
  
  std::cout << "mat.determinant() = \n" << mat.determinant() << std::endl << std::endl;
  std::cout << "mat.inverse() = \n" << mat.inverse() << std::endl << std::endl;
  std::cout << "mat * mat.inverse() > 0.000001 = \n" << ( (mat * mat.inverse()) > 0.000001 ) << std::endl << std::endl;

  mat.slice(StaticIndexTuple<size/2,0>(), vec.transpose().shape() ) = vec.transpose();
  std::cout << "mat.slice(StaticIndexTuple<size/2,0>(), vec.transpose().shape() ) = vec.transpose() = \n" << mat << std::endl << std::endl;
  
}

void dynamic_matrix_test(int size){
  std::cout << std::endl << "---------------------------------" << std::endl << "Dynamic test: size = " << size << std::endl << "---------------------------------" << std::endl << std::endl;

  DynamicMatrix<double> mat;
  mat.resize(size,size);
  mat.for_all_indices([&](typename decltype(mat)::Index idx){
    int i = idx.template get<0>(), j = idx.template get<1>();
    mat(idx) = 2+i/(j+1);
  });
  
  
  DynamicMatrix<double>::Copy c;
  c = mat;
  
  std::cout << "operator= test \n" << c << std::endl;
  
  DynamicMatrix<double> vec;
  vec.resize(size,1);
  vec.for_all_indices([&](decltype(vec)::Index idx){ int i = idx.get<0>(); vec(idx) = i + 1; });
  
  std::cout << vec.shape() << std::endl;
  
  std::cout << "mat = \n" << mat << std::endl << std::endl;
  
  std::cout << "mat[0] = " << mat[0] << std::endl << std::endl;
  std::cout << "mat.transpose()[0] = " << mat.transpose()[0] << std::endl << std::endl;
  
  std::cout << "vec = \n" << vec << std::endl << std::endl;
  std::cout << "vec.transpose() = " << vec.transpose() << std::endl << std::endl;
  
  std::cout << "mat * vec = \n" << mat * vec << std::endl << std::endl;
  std::cout << "mat.transpose() * mat = \n" << mat.transpose() * mat << std::endl << std::endl;
  std::cout << "vec.transpose() * mat = " << vec.transpose() * mat << std::endl << std::endl;
  std::cout << "vec * vec.transpose() = \n" << vec * vec.transpose() << std::endl << std::endl;
  
  std::cout << "vec.transpose() * vec = " << vec.transpose() * vec << std::endl << std::endl;
  
  float scalar = (vec.transpose() * vec)[0][0];
  std::cout << "float(vec.transpose() * vec) = " << scalar << std::endl << std::endl;
  
  std::cout << "mat.determinant() = \n" << mat.determinant() << std::endl << std::endl;
  std::cout << "mat.inverse() = \n" << mat.inverse() << std::endl << std::endl;
  std::cout << "mat * mat.inverse() > 0.000001 = \n" << ( (mat * mat.inverse()) > 0.000001 ) << std::endl << std::endl;
  
  mat.slice(DynamicIndexTuple<2>(size/2,0), vec.transpose().shape() ) = vec.transpose();
  std::cout << "mat.slice(DynamicIndexTuple<2>(size/2,0), vec.transpose().shape() ) = vec.transpose() = \n" << mat << std::endl << std::endl;

}


int main(){  
  
  static_matrix_test<1>();
  static_matrix_test<2>();
  static_matrix_test<3>();
  static_matrix_test<4>();
  static_matrix_test<8>();
  static_matrix_test<16>();
  
  dynamic_matrix_test(1);
  dynamic_matrix_test(2);
  dynamic_matrix_test(3);
  dynamic_matrix_test(4);
  dynamic_matrix_test(8);
  dynamic_matrix_test(16);
  
  return 0;
}

