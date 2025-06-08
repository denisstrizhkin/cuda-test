#pragma once

#include "cuda_err.cuh"
#include "cuda_ptr.cuh"
#include "rand.h"
#include <algorithm>
#include <cmath>
#include <cuda/std/limits>
#include <limits>
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
  for (size_t tI = 0; tI < tN; tI++) {
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
    for (size_t i = 0; i < WARP_SIZE; i++) {
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
  const auto row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n) {
    auto max_val = ::cuda::std::numeric_limits<T>::lowest();
    for (size_t i = 0; i < m; ++i) {
      max_val = max(max_val, mA[row * m + i]);
    }
    auto sum_exp = static_cast<T>(0);
    for (size_t i = 0; i < m; ++i) {
      const auto val = exp(mA[row * m + i] - max_val);
      mB[row * m + i] = val;
      sum_exp += val;
    }
    for (size_t i = 0; i < m; ++i) {
      mB[row * m + i] /= sum_exp;
    }
  }
}

template <typename T>
__global__ void mat_flash_attention(
    const T *__restrict__ Q_hbm, const T *__restrict__ K_hbm,
    const T *__restrict__ V_hbm, T *__restrict__ O_hbm,
    T *__restrict__ l_hbm, // l vector (log-sum-exp normalization)
    T *__restrict__ m_hbm, // m vector (row-wise max)
    const size_t N, const size_t D, const size_t Tr, const size_t Tc,
    const size_t Br, // Row block size for Q_i, O_i
    const size_t Bc  // Column block size for K_j, V_j
) {
  const size_t global_q_row = blockIdx.x * Br + threadIdx.x;
  if (global_q_row >= N) {
    return;
  }
  extern __shared__ T smem[];
  T *sQ = &smem[0];
  T *sK = &smem[Br * D];
  T *sV = &smem[Br * D + Bc * D];
  T *sS = &smem[Br * D + Bc * D * 2];
  for (size_t col = 0; col < D; col++) {
    sQ[threadIdx.x * D + col] = Q_hbm[global_q_row * D + col];
  }
  T current_l = l_hbm[global_q_row];
  T current_m = m_hbm[global_q_row];
  for (size_t j = 0; j < Tc; j++) {
    for (size_t row = 0; row < Bc; row++) {
      for (size_t col = 0; col < D; col++) {
        const size_t global_row = (j * Bc + row);
        const size_t global_ij = global_row * D + col;
        const size_t ij = row * D + col;
        if (global_row < N) {
          sK[ij] = K_hbm[global_ij];
          sV[ij] = V_hbm[global_ij];
        } else {
          sK[ij] = static_cast<T>(0);
          sV[ij] = static_cast<T>(0);
        }
      }
    }
    __syncthreads();
    T row_m = ::cuda::std::numeric_limits<T>::lowest();
    for (size_t col = 0; col < Bc; col++) {
      T sum = static_cast<T>(0);
      for (size_t d = 0; d < D; d++) {
        sum += sQ[blockIdx.x * D + d] * sK[d * Bc + col];
      }
      sS[threadIdx.x * Bc + col] = sum;
      row_m = max(row_m, sum);
    }
    T row_l = static_cast<T>(0);
    for (size_t col = 0; col < Bc; col++) {
      const size_t ij = threadIdx.x * Bc + col;
      sS[ij] = exp(sS[ij] - row_m);
      row_l += sS[ij];
    }
    const T new_m = max(row_m, current_m);
    const T new_l =
        exp(current_m - new_m) * current_l + exp(row_m - new_m) * row_l;
    for (size_t d = 0; d < D; d++) {
      T pv = static_cast<T>(0);
      for (size_t col = 0; col < Bc; col++) {
        pv += sS[blockIdx.x * D + col] * sV[col * D + d];
      }
      O_hbm[global_q_row * D + d] =
          (1 / new_l) *
          ((current_l * exp(current_m - new_m) * O_hbm[global_q_row * D + d]) +
           (exp(row_m - new_m) * pv));
    }
    l_hbm[global_q_row] = new_l;
    m_hbm[global_q_row] = new_m;
    __syncthreads();
  }
}

namespace Mat {

template <typename T, size_t N, size_t M> class Mat {
public:
  explicit Mat() : data_(CudaPtr<T>(N * M)) {}
  explicit Mat(const std::vector<T> &v) : data_(CudaPtr<T>(N * M)) {
    data_.copy_from_host_ptr(v.data(), size_bytes());
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
    if constexpr (N != 0 && M != 0 && K != 0) {
      dim3 dimBlock(WARP_SIZE, WARP_SIZE);
      dim3 dimGrid((K + dimBlock.x - 1) / dimBlock.x,
                   (N + dimBlock.y - 1) / dimBlock.y);
      CUDA_CHECK(cudaMemPrefetchAsync(data(), size_bytes(), 0));
      CUDA_CHECK(cudaMemPrefetchAsync(other.data(), other.size_bytes(), 0));
      mat_mul_naive<<<dimGrid, dimBlock>>>(data(), other.data(), result.data(),
                                           N, M, K);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    return result;
  };

  template <size_t K> Mat<T, N, K> dot_shared(const Mat<T, M, K> &other) const {
    Mat<T, N, K> result;
    if constexpr (N != 0 && M != 0 && K != 0) {
      dim3 dimBlock(WARP_SIZE, WARP_SIZE);
      dim3 dimGrid((K + dimBlock.x - 1) / dimBlock.x,
                   (N + dimBlock.y - 1) / dimBlock.y);
      CUDA_CHECK(cudaMemPrefetchAsync(data(), size_bytes(), 0));
      CUDA_CHECK(cudaMemPrefetchAsync(other.data(), other.size_bytes(), 0));
      mat_mul_shared<<<dimGrid, dimBlock>>>(data(), other.data(), result.data(),
                                            N, M, K);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    return result;
  };

  template <size_t K>
  Mat<T, N, K>
  dot_shared_with_warp_intrinsics(const Mat<T, M, K> &other) const {
    Mat<T, N, K> result;
    if constexpr (N != 0 && M != 0 && K != 0) {
      dim3 dimBlock(WARP_SIZE, WARP_SIZE);
      dim3 dimGrid((K + dimBlock.x - 1) / dimBlock.x,
                   (N + dimBlock.y - 1) / dimBlock.y);
      CUDA_CHECK(cudaMemPrefetchAsync(data(), size_bytes(), 0));
      CUDA_CHECK(cudaMemPrefetchAsync(other.data(), other.size_bytes(), 0));
      // mat_mul_shared_with_warp_intrinsics<<<dimGrid, dimBlock>>>(
      //     data(), other.data(), result.data(), N, M, K);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    return result;
  };

  Mat<T, M, N> transpose() const {
    Mat<T, M, N> result;
    if constexpr (N != 0 && M != 0) {
      dim3 dimBlock(WARP_SIZE, WARP_SIZE);
      dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                   (M + dimBlock.y - 1) / dimBlock.y);
      CUDA_CHECK(cudaMemPrefetchAsync(data(), size_bytes(), 0));
      mat_transpose<<<dimGrid, dimBlock>>>(data(), result.data(), N, M);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    return result;
  }

  Mat<T, N, M> softmax() const {
    Mat<T, N, M> result;
    if constexpr (N != 0 && M != 0) {
      dim3 dimGrid(N, 1, 1);
      dim3 dimBlock(1, 1, 1);
      CUDA_CHECK(cudaMemPrefetchAsync(data(), size_bytes(), 0));
      mat_softmax<<<dimGrid, dimBlock>>>(data(), result.data(), N, M);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    return result;
  }

  static Mat<T, N, M> zeros() {
    const std::vector<T> v(N * M, 0);
    return Mat<T, N, M>(v);
  }

  static Mat<T, N, M> random() {
    Mat<T, N, M> mat;
    if constexpr (N != 0 && M != 0) {
      for (size_t i = 0; i < N * M; ++i) {
        mat.data_[i] = Random::getInstance().next();
      }
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

  static Mat<T, N, M> flash_attention(const Mat<T, N, M> &mQ,
                                      const Mat<T, N, M> &mK,
                                      const Mat<T, N, M> &mV) {
    Mat<T, N, M> result;
    if constexpr (N != 0 && M != 0) {
      const size_t seq_len = N;
      const size_t head_dim = M;
      const size_t Br = 64;
      const size_t Bc = 64;
      const size_t Tr = (seq_len + Br - 1) / Br;
      const size_t Tc = (seq_len + Bc - 1) / Bc;
      CudaPtr<T> l_ptr(seq_len);
      const std::vector vl(seq_len, static_cast<T>(0));
      l_ptr.copy_from_host_ptr(vl.data(), vl.size() * sizeof(T));
      CudaPtr<T> m_ptr(seq_len);
      const std::vector vm(seq_len, std::numeric_limits<T>::lowest());
      m_ptr.copy_from_host_ptr(vm.data(), vm.size() * sizeof(T));
      dim3 dimGrid(Tr, 1, 1);
      dim3 dimBlock(Br, 1, 1);
      const size_t shared_mem_size =
          (Br * head_dim + Bc * head_dim + Bc * head_dim + Br * Bc) * sizeof(T);
      std::cout << "about to start flash attention\n";
      mat_flash_attention<<<dimGrid, dimBlock, shared_mem_size>>>(
          mQ.data(), mK.data(), mV.data(), result.data(), l_ptr.get(),
          m_ptr.get(), N, M, Tr, Tc, Br, Bc);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    return result;
  }

private:
  CudaPtr<T> data_;
};

}; // namespace Mat
