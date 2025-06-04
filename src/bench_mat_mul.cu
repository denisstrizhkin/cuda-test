#include "mat.cuh"
#include "rand.h"

#include <cassert>
#include <chrono>
#include <vector>

void print_bench_header(const std::string &name, const size_t n, const size_t m,
                        const size_t k) {
  std::cout << "\n=== bench " << name << " (" << n << "x" << m << ") by (" << m
            << "x" << k << ") mul ===\n";
}

void print_bench_footer(const float millis) {
  std::cout << "=== finished in " << millis << "ms ===\n";
}

template <typename T, size_t N, size_t M, size_t K, typename DotProductFunc>
void bench_mul(const std::string &test_name, DotProductFunc dot_func) {
  print_bench_header(test_name, N, M, K);
  const auto a = Mat::Mat<T, N, M>::random();
  const auto b = Mat::Mat<T, M, K>::random();
  cudaEvent_t start_event, stop_event;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));
  CUDA_CHECK(cudaEventRecord(start_event));
  const auto c = (a.*dot_func)(b);
  CUDA_CHECK(cudaEventRecord(stop_event));
  CUDA_CHECK(cudaEventSynchronize(stop_event));
  float millis = 0;
  CUDA_CHECK(cudaEventElapsedTime(&millis, start_event, stop_event));
  print_bench_footer(millis);
}

template <typename T, size_t N, size_t M, size_t K> void bench_mul_naive() {
  bench_mul<T, N, M, K>("naive", &Mat::Mat<T, N, M>::template dot_naive<K>);
}

template <typename T, size_t N, size_t M, size_t K> void bench_mul_shared() {
  bench_mul<T, N, M, K>("shared", &Mat::Mat<T, N, M>::template dot_shared<K>);
}

template <typename T, size_t N, size_t M, size_t K>
void bench_mul_shared_with_warp_intrinsics() {
  bench_mul<T, N, M, K>(
      "shared with warp intrinsics",
      &Mat::Mat<T, N, M>::template dot_shared_with_warp_intrinsics<K>);
}

template <typename T, size_t N, size_t M, size_t K> void bench_mul_cpu() {
  print_bench_header("cpu", N, M, K);
  std::vector<T> a(N * M);
  for (auto i = 0; i < a.size(); i++) {
    a[i] = Random::getInstance().next();
  }
  std::vector<T> b(M * K);
  for (auto i = 0; i < b.size(); i++) {
    b[i] = Random::getInstance().next();
  }
  std::vector<T> c(N * K);
  const auto start_time = std::chrono::high_resolution_clock::now();
  for (auto i = 0; i < N; i++) {
    for (auto j = 0; j < K; j++) {
      T sum = 0;
      for (auto k = 0; k < M; k++) {
        sum += a[i * M + k] * b[k * K + j];
      }
      c[i * K + j] = sum;
    }
  }
  const auto end_time = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<float, std::milli> duration =
      end_time - start_time;
  const auto millis = duration.count();
  print_bench_footer(millis);
}

int main(void) {
  cudaDeviceProp deviceProp;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
  std::cout << "=== Device Info ===\n"
            << "Max shared memory per block: "
            << deviceProp.sharedMemPerBlock / 1024.0 << "KB\n"
            << "===\n";
  std::cout << "\n--- Benchmarking CPU Multiplication ---\n";
  bench_mul_cpu<float, 8, 16, 32>();
  bench_mul_cpu<float, 64, 128, 256>();
  // bench_mul_cpu<float, 1024, 2048, 4096>(); // too slow anyway
  std::cout << "\n--- Benchmarking Naive Multiplication ---\n";
  bench_mul_naive<float, 8, 16, 32>();
  bench_mul_naive<float, 64, 128, 256>();
  bench_mul_naive<float, 1024, 2048, 4096>();
  std::cout << "\n--- Benchmarking Shared Multiplication ---\n";
  bench_mul_shared<float, 8, 16, 32>();
  bench_mul_shared<float, 64, 128, 256>();
  bench_mul_shared<float, 1024, 2048, 4096>();
  std::cout
      << "\n--- Benchmarking Shared Multiplication With Warp Intrinsics ---\n";
  bench_mul_shared_with_warp_intrinsics<float, 8, 16, 32>();
  bench_mul_shared_with_warp_intrinsics<float, 64, 128, 256>();
  bench_mul_shared_with_warp_intrinsics<float, 1024, 2048, 4096>();
  return EXIT_SUCCESS;
}
