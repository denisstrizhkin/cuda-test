#include "cuda_err.cuh"
#include "mat.cuh"
#include "rand.h"

const size_t WARP_SIZE = 32;

template <typename T>
__global__ void mat_mul(const T *const __restrict__ mA,
                        const T *const __restrict__ mB,
                        T *const __restrict__ mC, const size_t A_rows,
                        const size_t A_cols_B_rows, const size_t B_cols) {
  const auto row = blockIdx.y * blockDim.y + threadIdx.y;
  const auto col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < A_rows && col < B_cols) {
    T sum = 0;
    for (auto i = 0; i < A_cols_B_rows; i++) {
      sum += mA[row * A_cols_B_rows + i] * mB[i * B_cols + col];
    }
    mC[row * B_cols + col] = sum;
  }
}

// --- Mat Class Implementations ---

// Default constructor: Uses compile-time N and M for CudaPtr allocation
template <typename T, size_t N, size_t M>
Mat::Mat<T, N, M>::Mat() : data_(CudaPtr<T>(N * M)) {}

// Copy constructor: Uses compile-time N and M from 'this' type
template <typename T, size_t N, size_t M>
Mat::Mat<T, N, M>::Mat(const Mat &other) : data_(CudaPtr<T>(N * M)) {
  data_.copy_from_cuda_ptr(other.data_);
}

// Move constructor: Transfers ownership
template <typename T, size_t N, size_t M>
Mat::Mat<T, N, M>::Mat(Mat &&other) noexcept : data_(other.data_) {}

// at() for mutable access: Uses compile-time M for column stride
template <typename T, size_t N, size_t M>
T &Mat::Mat<T, N, M>::at(const size_t i, const size_t j) noexcept {
  return data_[i * M + j];
}

// at() for constant access: Uses compile-time M for column stride
template <typename T, size_t N, size_t M>
const T &Mat::Mat<T, N, M>::at(const size_t i, const size_t j) const noexcept {
  return data_[i * M + j];
}

// dot() for matrix multiplication: Computes 'this' (N x M) * 'other' (M x K) =
// 'result' (N x K).
template <typename T, size_t N, size_t M>
template <size_t K>
Mat::Mat<T, N, K> Mat::Mat<T, N, M>::dot(const Mat<T, M, K> &other) const {
  Mat<T, N, K> result;
  dim3 dimBlock(WARP_SIZE, WARP_SIZE);
  dim3 dimGrid((K + dimBlock.x - 1) / dimBlock.x,
               (N + dimBlock.y - 1) / dimBlock.y);
  mat_mul<<<dimGrid, dimBlock>>>(data_.get(), other.data_.get(),
                                 result.data_.get(), N, M, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return result;
}

// random() factory function: Uses compile-time N and M for the created matrix.
template <typename T, size_t N, size_t M> Mat::Mat<T, N, M> Mat::random() {
  Mat<T, N, M> mat;
  for (size_t i = 0; i < N * M; ++i) {
    mat.data_[i] = Random::getInstance().next();
  }
  return mat;
}
