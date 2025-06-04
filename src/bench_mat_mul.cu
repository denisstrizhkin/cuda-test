#include "mat.cuh"

#include <cassert>

template <typename T, size_t N, size_t M, size_t K, typename DotProductFunc>
void bench_mul(const std::string &test_name, DotProductFunc dot_func) {
  std::cout << "\n=== bench " << test_name << " (" << N << "x" << M << ") by ("
            << M << "x" << K << ") mul ===\n";
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
  std::cout << "=== finished in " << millis << "ms ===\n";
}

template <typename T, size_t N, size_t M, size_t K> void bench_mul_naive() {
  bench_mul<T, N, M, K>("naive", &Mat::Mat<T, N, M>::template dot_naive<K>);
}

template <typename T, size_t N, size_t M, size_t K> void bench_mul_shared() {
  bench_mul<T, N, M, K>("shared", &Mat::Mat<T, N, M>::template dot_shared<K>);
}

int main(void) {
  cudaDeviceProp deviceProp;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
  std::cout << "=== Device Info ===\n"
            << "Max shared memory per block: "
            << deviceProp.sharedMemPerBlock / 1024.0 << "KB\n"
            << "===\n";
  std::cout << "\n--- Benchmarking Naive Multiplication ---\n";
  bench_mul_naive<float, 8, 16, 32>();
  bench_mul_naive<float, 64, 128, 256>();
  bench_mul_naive<float, 1024, 2048, 4096>();
  std::cout << "\n--- Benchmarking Shared Multiplication ---\n";
  bench_mul_shared<float, 8, 16, 32>();
  bench_mul_shared<float, 64, 128, 256>();
  bench_mul_shared<float, 1024, 2048, 4096>();
  return EXIT_SUCCESS;
}
