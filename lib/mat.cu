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

template <typename T>
Mat<T>::Mat(const size_t n, const size_t m) noexcept
    : n_(n), m_(m), data_(CudaPtr<T>(n * m)) {}

template <typename T>
Mat<T>::Mat(const Mat &other) noexcept : Mat(other.n_, other.m_) {
  data_.copy_from_cuda_ptr(other.data_);
}

template <typename T> Mat<T> &Mat<T>::operator=(const Mat &other) noexcept {
  if (this == &other) {
    return *this;
  }
  n_ = other.n_;
  m_ = other.m_;
  if (data_.size() < other.size()) {
    data_ = CudaPtr<T>(n_ * m_);
  }
  data_.copy_from_cuda_ptr(other.data_);
  return *this;
}

template <typename T>
Mat<T>::Mat(Mat &&other) noexcept
    : n_(other.n_), m_(other.m_), data_(std::move(other.data_)) {
  other.n_ = 0;
  other.m_ = 0;
}

template <typename T> Mat<T> &Mat<T>::operator=(Mat &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  n_ = other.n_;
  m_ = other.m_;
  data_ = std::move(other.data_);
  other.n_ = 0;
  other.m_ = 0;
  return *this;
}

template <typename T> T &Mat<T>::at(const size_t i, const size_t j) noexcept {
  return data_[i * m_ + j];
}

template <typename T>
const T &Mat<T>::at(const size_t i, const size_t j) const noexcept {
  return data_[i * m_ + j];
}

template <typename T> Mat<T> Mat<T>::dot(const Mat &other) const noexcept {
  if (m_ != other.n_) {
    // You might want to throw an exception here or handle this error more
    // robustly For simplicity, returning a default-constructed Mat or logging
    // an error.
    fprintf(stderr,
            "Error: Incompatible dimensions for matrix multiplication! (%zu x "
            "%zu) dot (%zu x %zu)\n",
            n_, m_, other.n_, other.m_);
    return Mat<T>(0, 0); // Return an empty matrix
  }
  const auto A_rows = n_;
  const auto A_cols_B_rows = m_;
  const auto B_cols = other.m_;
  auto mC = Mat<T>(A_rows, B_cols);
  dim3 dimBlock(WARP_SIZE, WARP_SIZE);
  dim3 dimGrid((B_cols + dimBlock.x - 1) / dimBlock.x,
               (A_rows + dimBlock.y - 1) / dimBlock.y);
  mat_mul<<<dimGrid, dimBlock>>>(data_.get(), other.data_.get(), mC.data_.get(),
                                 A_rows, A_cols_B_rows, B_cols);
  return mC;
}

template <typename T>
Mat<T> Mat<T>::random(const size_t n, const size_t m) noexcept {
  Mat mat(n, m);
  std::vector<T> host_data(n * m);
  for (auto i = 0; i < n; i++) {
    for (auto j = 0; j < m; j++) {
      host_data[i * m + j] = Random::getInstance().next();
    }
  }
  mat.data_.copy_from_host_ptr(host_data.data(), n * m);
  return mat;
}
