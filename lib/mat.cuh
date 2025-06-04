#pragma once

#include "cuda_ptr.cuh"
#include "rand.h"


const size_t WARP_SIZE = 32;


template <typename T>
__global__ void mat_mul_naive(const T *const __restrict__ mA,
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

template <typename T>
__global__ void
mat_mul_shared(const T *const __restrict__ mA, const T *const __restrict__ mB,
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


namespace Mat {

template <typename T, size_t N, size_t M> class Mat {
public:
  explicit Mat() : data_(CudaPtr<T>(N * M)) {}
  ~Mat() noexcept = default;

  Mat(const Mat &other) : data_(CudaPtr<T>(N * M)) {
    data_.copy_from_cuda_ptr(other.data_);
  }
  Mat &operator=(const Mat &other) = delete;

  Mat(Mat &&other) noexcept : data_(std::move(other.data_)) {}
  Mat &operator=(Mat &&other) = delete;

  size_t rows() const noexcept { return N; }
  size_t cols() const noexcept { return M; }
  size_t size() const noexcept { return data_.size(); }
  size_t size_bytes() const noexcept { return data_.size_bytes(); }

  T *data() noexcept { return data_.get(); }
  const T *data() const noexcept { return data_.get(); }

  T &at(const size_t i, const size_t j) noexcept { return data_[i * M + j]; }
  const T &at(const size_t i, const size_t j) const noexcept { return data_[i * M + j]; }

  template <size_t K> Mat<T, N, K> dot_naive(const Mat<T, M, K> &other) const {
    Mat<T, N, K> result;
    if (N == 0 || K == 0) {
      return result;
    }
    dim3 dimBlock(WARP_SIZE, WARP_SIZE);
    dim3 dimGrid((K + dimBlock.x - 1) / dimBlock.x,
                 (N + dimBlock.y - 1) / dimBlock.y);
    mat_mul_naive<<<dimGrid, dimBlock>>>(data(), other.data(), result.data(), N,
                                         M, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
  };

  template <size_t K> Mat<T, N, K> dot_shared(const Mat<T, M, K> &other) const {
    Mat<T, N, K> result;
    if (N == 0 || K == 0) {
      return result;
    }
    dim3 dimBlock(WARP_SIZE, WARP_SIZE);
    dim3 dimGrid((K + dimBlock.x - 1) / dimBlock.x,
                 (N + dimBlock.y - 1) / dimBlock.y);
    mat_mul_naive<<<dimGrid, dimBlock>>>(data(), other.data(), result.data(), N,
                                         M, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
  };

  static Mat<T, N, M> random() {
    Mat<T, N, M> mat;
    for (size_t i = 0; i < N * M; ++i) {
      mat.data_[i] = Random::getInstance().next();
    }
    return mat;
  }

private:
  CudaPtr<T> data_;
};

}; // namespace Mat
