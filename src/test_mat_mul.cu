#include <cstdlib>
#include <iostream>

#include "mat.cuh"

template<typename T, size_t N, size_t M>
void print_mat(const Mat::Mat<T, N, M>& mat) {
  std::cout << "===\n";
  std::cout << "mat.rows(): " << mat.rows() << "\n";
  std::cout << "mat.cols(): " << mat.cols() << "\n";
  std::cout << "mat.size(): " << mat.size() << "\n";
  std::cout << "mat.size_bytes(): " << mat.size_bytes() << "\n";
  for (auto i = 0; i < mat.rows(); i++) {
    for (auto j = 0; j < mat.cols(); j++) {
      std::cout << mat.at(i, j);
    }
    std::cout << "\n";
  }
}

int main(void) {
  const auto m = Mat::random<double, 3, 2>();
  print_mat(m);
  return EXIT_SUCCESS;
}
