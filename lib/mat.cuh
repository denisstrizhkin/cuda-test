#pragma once

#include "cuda_err.cuh"
#include "cuda_ptr.cuh"
#include "rand.h"
#include <cuda/std/limits>
#include <vector>

const size_t WARP_SIZE = 32;

template <typename T>
__global__ void mat_mul_naive(const T *const __restrict__ mA,
                              const T *const __restrict__ mB,
                              T *const __restrict__ mC, const size_t n,
                              const size_t m, const size_t k) {
  const auto row = blockIdx.y * blockDim.y + threadIdx.y;
  const auto col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < k) {
    T sum = 0;
    for (auto i = 0; i < m; i++) {
      sum += mA[row * m + i] * mB[i * k + col];
    }
    mC[row * k + col] = sum;
  }
}

template <typename T>
__global__ void mat_mul_shared(const T *const __restrict__ mA,
                               const T *const __restrict__ mB,
                               T *const __restrict__ mC, const size_t n,
                               const size_t m, const size_t k) {
  __shared__ T tA[WARP_SIZE][WARP_SIZE];
  __shared__ T tB[WARP_SIZE][WARP_SIZE];
  const auto tN = (m + WARP_SIZE - 1) / WARP_SIZE;
  const auto row = blockIdx.y * blockDim.y + threadIdx.y;
  const auto col = blockIdx.x * blockDim.x + threadIdx.x;
  T sum = 0;
  for (auto tI = 0; tI < tN; tI++) {
    const auto a_row = row;
    const auto a_col = tI * WARP_SIZE + threadIdx.x;
    if (a_row < n && a_col < m) {
      tA[threadIdx.y][threadIdx.x] = mA[a_row * m + a_col];
    } else {
      tA[threadIdx.y][threadIdx.x] = 0.0;
    }
    const auto b_row = tI * WARP_SIZE + threadIdx.y;
    const auto b_col = col;
    if (b_row < m && b_col < k) {
      tB[threadIdx.y][threadIdx.x] = mB[b_row * k + b_col];
    } else {
      tB[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    for (auto i = 0; i < WARP_SIZE; i++) {
      sum += tA[threadIdx.y][i] * tB[i][threadIdx.x];
    }
    __syncthreads();
  }
  if (row < n && col < k) {
    mC[row * k + col] = sum;
  }
}

template <typename T>
__global__ void mat_transpose(const T *const __restrict__ mA,
                              T *const __restrict__ mB, const size_t n,
                              const size_t m) {
  const auto row = blockIdx.y * blockDim.y + threadIdx.y;
  const auto col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    mB[row * n + col] = mA[col * m + row];
  }
}

template <typename T>
__global__ void mat_softmax(const T *const __restrict__ mA,
                            T *const __restrict__ mB, const size_t n,
                            const size_t m) {
  const auto row = blockIdx.x;
  if (row < n) {
    auto max_val = ::cuda::std::numeric_limits<T>::lowest();
    for (auto i = 0; i < m; ++i) {
      max_val = max(max_val, mA[row * m + i]);
    }
    auto sum_exp = static_cast<T>(0);
    for (auto i = 0; i < m; ++i) {
      sum_exp += exp(mA[row * m + i] - max_val);
    }
    for (auto i = 0; i < m; ++i) {
      mB[row * m + i] = exp(mA[row * m + i] - max_val) / sum_exp;
    }
  }
}

namespace Mat {

template <typename T, size_t N, size_t M> class Mat {
public:
  explicit Mat() : data_(CudaPtr<T>(N * M)) {}
  explicit Mat(const std::vector<T> &v) : data_(CudaPtr<T>(N * M)) {
    data_.copy_from_host_ptr(v.data(), N * M);
  }
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
  const T &at(const size_t i, const size_t j) const noexcept {
    return data_[i * M + j];
  }

  template <size_t K> Mat<T, N, K> dot_naive(const Mat<T, M, K> &other) const {
    Mat<T, N, K> result;
    if (N == 0 || M == 0 || K == 0) {
      return result;
    }
    dim3 dimBlock(WARP_SIZE, WARP_SIZE);
    dim3 dimGrid((K + dimBlock.x - 1) / dimBlock.x,
                 (N + dimBlock.y - 1) / dimBlock.y);
    CUDA_CHECK(cudaMemPrefetchAsync(data(), size_bytes(), 0));
    CUDA_CHECK(cudaMemPrefetchAsync(other.data(), other.size_bytes(), 0));
    mat_mul_naive<<<dimGrid, dimBlock>>>(data(), other.data(), result.data(), N,
                                         M, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
  };

  template <size_t K> Mat<T, N, K> dot_shared(const Mat<T, M, K> &other) const {
    Mat<T, N, K> result;
    if (N == 0 || M == 0 || K == 0) {
      return result;
    }
    dim3 dimBlock(WARP_SIZE, WARP_SIZE);
    dim3 dimGrid((K + dimBlock.x - 1) / dimBlock.x,
                 (N + dimBlock.y - 1) / dimBlock.y);
    CUDA_CHECK(cudaMemPrefetchAsync(data(), size_bytes(), 0));
    CUDA_CHECK(cudaMemPrefetchAsync(other.data(), other.size_bytes(), 0));
    mat_mul_shared<<<dimGrid, dimBlock>>>(data(), other.data(), result.data(),
                                          N, M, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
  };

  template <size_t K>
  Mat<T, N, K>
  dot_shared_with_warp_intrinsics(const Mat<T, M, K> &other) const {
    Mat<T, N, K> result;
    if (N == 0 || M == 0 || K == 0) {
      return result;
    }
    dim3 dimBlock(WARP_SIZE, WARP_SIZE);
    dim3 dimGrid((K + dimBlock.x - 1) / dimBlock.x,
                 (N + dimBlock.y - 1) / dimBlock.y);
    CUDA_CHECK(cudaMemPrefetchAsync(data(), size_bytes(), 0));
    CUDA_CHECK(cudaMemPrefetchAsync(other.data(), other.size_bytes(), 0));
    // mat_mul_shared_with_warp_intrinsics<<<dimGrid, dimBlock>>>(
    //     data(), other.data(), result.data(), N, M, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
  };

  Mat<T, M, N> transpose() const {
    Mat<T, M, N> result;
    if (N == 0 || M == 0) {
      return result;
    }
    dim3 dimBlock(WARP_SIZE, WARP_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                 (M + dimBlock.y - 1) / dimBlock.y);
    CUDA_CHECK(cudaMemPrefetchAsync(data(), size_bytes(), 0));
    mat_transpose<<<dimGrid, dimBlock>>>(data(), result.data(), N, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
  }

  Mat<T, N, M> softmax() const {
    Mat<T, N, M> result;
    if (N == 0 || M == 0) {
      return result;
    }
    dim3 dimGrid(N, 1, 1);
    dim3 dimBlock(1, 1, 1);
    CUDA_CHECK(cudaMemPrefetchAsync(data(), size_bytes(), 0));
    mat_softmax<<<dimGrid, dimBlock>>>(data(), result.data(), N, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
  }

  static Mat<T, N, M> random() {
    Mat<T, N, M> mat;
    for (size_t i = 0; i < N * M; ++i) {
      mat.data_[i] = Random::getInstance().next();
    }
    return mat;
  }

  static Mat<T, N, M> attention(const Mat<T, N, M> &mQ, const Mat<T, N, M> &mK,
                                const Mat<T, N, M> &mV) {
    const auto mKT = mK.transpose();
    const auto s = mQ.dot_shared(mKT);
    const auto p = s.softmax();
    return p.dot_shared(mV);
  }

private:
  CudaPtr<T> data_;
};

}; // namespace Mat
