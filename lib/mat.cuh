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
__global__ void mat_mul_shared(const T *const __restrict__ mA,
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

namespace Mat {

/**
 * @brief A templated class representing a fixed-size matrix stored on a CUDA
 * device.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double, int).
 * @tparam N The number of rows in the matrix (compile-time constant).
 * @tparam M The number of columns in the matrix (compile-time constant).
 */
template <typename T, size_t N, size_t M> class Mat {
public:
  /**
   * @brief Constructs an empty (uninitialized) matrix.
   *
   * Initializes the underlying CudaPtr. The device memory for the matrix
   * is allocated but its contents are not set.
   */
  explicit Mat() : data_(CudaPtr<T>(N * M)) {}

  /**
   * @brief Destroys the matrix.
   *
   * The default destructor ensures that the underlying CudaPtr's destructor
   * is called, which should deallocate the device memory automatically.
   */
  ~Mat() noexcept = default;

  /**
   * @brief Copy constructor.
   *
   * Performs a deep copy of the `other` matrix, allocating new device memory
   * and copying the data from the source matrix.
   *
   * @param other The matrix to copy from.
   */
  Mat(const Mat &other) : data_(CudaPtr<T>(N * M)) {
    data_.copy_from_cuda_ptr(other.data_);
  }

  /**
   * @brief Deleted copy assignment operator.
   */
  Mat &operator=(const Mat &other) = delete;

  /**
   * @brief Move constructor.
   *
   * Transfers ownership of the device memory from `other` to the newly
   * constructed matrix. The `other` matrix will be left in a valid but
   * unspecified state (typically empty).
   *
   * @param other The matrix to move resources from.
   */
  Mat(Mat &&other) noexcept : data_(std::move(other.data_)) {}

  /**
   * @brief Deleted move assignment operator.
   *
   */
  Mat &operator=(Mat &&other) = delete;

  /**
   * @brief Returns the number of rows in the matrix.
   *
   * @return The compile-time constant number of rows, N.
   */
  size_t rows() const noexcept { return N; }

  /**
   * @brief Returns the number of columns in the matrix.
   *
   * @return The compile-time constant number of columns, M.
   */
  size_t cols() const noexcept { return M; }

  /**
   * @brief Returns the total number of elements in the matrix.
   *
   * This is equivalent to `rows() * cols()`.
   *
   * @return The total number of elements (N * M).
   */
  size_t size() const noexcept { return data_.size(); }

  /**
   * @brief Returns the total size of the matrix in bytes.
   *
   * This is equivalent to `rows() * cols() * sizeof(T)`.
   *
   * @return The size of the matrix data in bytes.
   */
  size_t size_bytes() const noexcept { return data_.size_bytes(); }

  const T *data() const noexcept { return data_.get(); }
  T *data() noexcept { return data_.get(); }

  /**
   * @brief Provides mutable access to an element at the specified row and
   * column.
   *
   * @param i The row index (0-based).
   * @param j The column index (0-based).
   * @return A mutable reference to the element at `(i, j)`.
   */
  T &at(const size_t i, const size_t j) noexcept { return data_[i * M + j]; }

  /**
   * @brief Provides constant access to an element at the specified row and
   * column.
   *
   * @param i The row index (0-based).
   * @param j The column index (0-based).
   * @return A constant reference to the element at `(i, j)`.
   */
  const T &at(const size_t i, const size_t j) const noexcept {
    return data_[i * M + j];
  }

  /**
   * @brief Performs matrix multiplication: `result = this * other`.
   *
   * This function computes the dot product (matrix multiplication) of the
   * current matrix (`this`) with `other`. It launches a CUDA kernel to perform
   * the computation on the device and returns a new `Mat` object with the
   * result.
   *
   * @param other The right-hand side matrix for the multiplication.
   * @return A new `Mat` object representing the product of `this` and `other`.
   */
  template <size_t K> Mat<T, N, K> dot_naive(const Mat<T, M, K> &other) const {
    Mat<T, N, K> result;
    if (N == 0 || K == 0) {
      return result;
    }
    dim3 dimBlock(WARP_SIZE, WARP_SIZE);
    dim3 dimGrid((K + dimBlock.x - 1) / dimBlock.x,
                 (N + dimBlock.y - 1) / dimBlock.y);
    mat_mul_naive<<<dimGrid, dimBlock>>>(data(), other.data(), result.data(), N, M,
                                   K);
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
  CudaPtr<T>
      data_; ///< Pointer to the device memory storing the matrix elements.
};

}; // namespace Mat
