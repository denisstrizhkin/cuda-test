#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "mat.cuh"

template <typename T, size_t N, size_t M>
void print_mat(const Mat::Mat<T, N, M> &mat) {
  std::stringstream oss;
  oss << "===\n";
  oss << "mat.rows(): " << mat.rows() << "\n";
  oss << "mat.cols(): " << mat.cols() << "\n";
  oss << "mat.size(): " << mat.size() << "\n";
  oss << "mat.size_bytes(): " << mat.size_bytes() << "\n";
  for (auto i = 0; i < mat.rows(); i++) {
    for (auto j = 0; j < mat.cols(); j++) {
      oss << mat.at(i, j) << " ";
    }
    oss << "\n";
  }
  oss << "===\n";
  std::cout << oss.str();
}

void test_random() {
  std::cout << "\n=== test random ===\n";
  const auto m = Mat::Mat<double, 3, 2>::random();
  assert(m.rows() == 3);
  assert(m.cols() == 2);
  assert(m.size() == 6);
  assert(m.size_bytes() == m.size() * sizeof(double));
  print_mat(m);
  std::cout << "=== OK ===\n";
}

template <typename T, size_t N, size_t M, size_t K> void test_mul_naive() {
  std::cout << "\n=== test (" << N << "x" << M << ") by (" << M << "x" << K
            << ") mul ===\n";
  const auto a = Mat::Mat<double, N, M>::random();
  const auto b = Mat::Mat<double, M, K>::random();
  const auto c = a.dot_naive(b);
  if (N < 5 && M < 5 && K < 5) {
    print_mat(a);
    print_mat(b);
    print_mat(c);
  }
  assert(c.size() == N * K);
  assert(c.size_bytes() == N * K * sizeof(T));
  assert(c.rows() == N);
  assert(c.cols() == K);
  for (auto i = 0; i < a.rows(); i++) {
    for (auto k = 0; k < b.cols(); k++) {
      auto expected = 0.0;
      for (auto j = 0; j < a.cols(); j++) {
        // std::cout << a.at(i, j) << " " << b.at(j, k) << "\n";
        expected += a.at(i, j) * b.at(j, k);
      }
      const auto got = c.at(i, k);
      const auto diff = std::abs(expected - got);
      if (diff > 1e-12) {
        std::cout << "at (" << i << "," << k << ") expected: " << expected
                  << ", got: " << got << "\n";
        assert(false);
      }
    }
  }
  std::cout << "=== OK ===\n";
}

int main(void) {
  test_random();
  test_mul_naive<double, 0, 0, 0>();
  test_mul_naive<double, 1, 1, 1>();
  test_mul_naive<double, 3, 2, 4>();
  test_mul_naive<double, 64, 128, 256>();
  test_mul_naive<double, 2048, 64, 4096>();
  return EXIT_SUCCESS;
}
