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

template <typename T, size_t N, size_t M, size_t K, typename DotProductFunc>
void test_mul(const std::string &test_name, DotProductFunc dot_func) {
  std::cout << "\n=== test " << test_name << " (" << N << "x" << M << ") by ("
            << M << "x" << K << ") mul ===\n";
  const auto a = Mat::Mat<T, N, M>::random();
  const auto b = Mat::Mat<T, M, K>::random();
  const auto c = (a.*dot_func)(b);
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

template <typename T, size_t N, size_t M, size_t K> void test_mul_naive() {
  test_mul<T, N, M, K>("naive", &Mat::Mat<T, N, M>::template dot_naive<K>);
}

template <typename T, size_t N, size_t M, size_t K> void test_mul_shared() {
  test_mul<T, N, M, K>("shared", &Mat::Mat<T, N, M>::template dot_shared<K>);
}

template <typename T, size_t N, size_t M, size_t K>
void test_mul_shared_with_warp_intrinsics() {
  test_mul<T, N, M, K>(
      "shared with warp intrinsics",
      &Mat::Mat<T, N, M>::template dot_shared_with_warp_intrinsics<K>);
}

int main(void) {
  test_random();
  std::cout << "\n--- Testing Naive Multiplication ---\n";
  test_mul_naive<double, 0, 0, 0>();
  test_mul_naive<double, 1, 1, 1>();
  test_mul_naive<double, 3, 2, 4>();
  test_mul_naive<double, 64, 128, 256>();
  test_mul_naive<double, 256, 512, 1024>();
  std::cout << "\n--- Testing Shared Multiplication ---\n";
  test_mul_shared<double, 0, 0, 0>();
  test_mul_shared<double, 1, 1, 1>();
  test_mul_shared<double, 3, 2, 4>();
  test_mul_shared<double, 64, 128, 256>();
  test_mul_shared<double, 256, 512, 1024>();
  std::cout << "\n--- Testing Shared Multiplication With Warp Intrinsics ---\n";
  // test_mul_shared_with_warp_intrinsics<double, 0, 0, 0>();
  // test_mul_shared_with_warp_intrinsics<double, 1, 1, 1>();
  // test_mul_shared_with_warp_intrinsics<double, 3, 2, 4>();
  // test_mul_shared_with_warp_intrinsics<double, 64, 128, 256>();
  // test_mul_shared_with_warp_intrinsics<double, 256, 512, 1024>();
  return EXIT_SUCCESS;
}
