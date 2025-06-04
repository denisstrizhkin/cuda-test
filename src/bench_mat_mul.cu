#include "mat.cuh"

#include <cassert>

template <typename T, size_t N, size_t M, size_t K, typename DotProductFunc>
void bench_mul(const std::string &test_name, DotProductFunc dot_func) {
  std::cout << "\n=== bench " << test_name << " (" << N << "x" << M << ") by ("
            << M << "x" << K << ") mul ===\n";
  const auto a = Mat::Mat<T, N, M>::random();
  const auto b = Mat::Mat<T, M, K>::random();
  const auto c = (a.*dot_func)(b);
  assert(c.size() == N * K);
  assert(c.size_bytes() == N * K * sizeof(T));
  assert(c.rows() == N);
  assert(c.cols() == K);
  std::cout << "=== FINISHED ===\n";
}

template <typename T, size_t N, size_t M, size_t K> void bench_mul_naive() {
  bench_mul<T, N, M, K>("naive", &Mat::Mat<T, N, M>::template dot_naive<K>);
}

template <typename T, size_t N, size_t M, size_t K> void bench_mul_shared() {
  bench_mul<T, N, M, K>("shared", &Mat::Mat<T, N, M>::template dot_shared<K>);
}

int main(void) {
  std::cout << "\n--- Benchmarking Naive Multiplication ---\n";
  bench_mul_naive<double, 64, 128, 256>();
  bench_mul_naive<double, 256, 512, 1024>();
  bench_mul_naive<double, 2048, 4096, 8192>();
  std::cout << "\n--- Benchmarking Shared Multiplication ---\n";
  bench_mul_shared<double, 64, 128, 256>();
  bench_mul_shared<double, 256, 512, 1024>();
  bench_mul_shared<double, 2048, 4096, 8192>();
  return EXIT_SUCCESS;
}
