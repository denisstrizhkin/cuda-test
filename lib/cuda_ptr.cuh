#pragma once

#include "cuda_err.cuh"

template <typename T> class CudaPtr {
public:
  explicit CudaPtr(const size_t n) : ptr_(nullptr), size_bytes_(n * sizeof(T)) {
    if (n == 0) {
      return;
    }
    CUDA_CHECK(cudaMallocManaged(&ptr_, size_bytes_));
  }

  ~CudaPtr() noexcept {
    if (ptr_ != nullptr) {
      try {
        CUDA_CHECK(cudaFree(ptr_))
      } catch (const CudaException &e) {
      }
      ptr_ = nullptr;
    }
  }

  CudaPtr(const CudaPtr &) = delete;
  CudaPtr &operator=(const CudaPtr &) = delete;

  CudaPtr(CudaPtr &&other) noexcept
      : ptr_(other.ptr_), size_bytes_(other.size_bytes_) {
    other.ptr_ = nullptr;
    other.size_bytes_ = 0;
  }

  CudaPtr &operator=(CudaPtr &&other) {
    if (this != &other) {
      if (ptr_ != nullptr) {
        CUDA_CHECK(cudaFree(ptr_));
      }
      ptr_ = other.ptr_;
      size_bytes_ = other.size_bytes_;
      other.ptr_ = nullptr;
      other.size_bytes_ = 0;
    }
    return *this;
  }

  T *get() noexcept { return ptr_; }
  const T *get() const noexcept { return ptr_; }

  T &operator*() const noexcept { return *ptr_; }
  T *operator->() const noexcept { return ptr_; }

  T &operator[](const size_t i) noexcept { return ptr_[i]; }
  const T &operator[](const size_t i) const noexcept { return ptr_[i]; }

  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  size_t size() const noexcept { return size_bytes_ / sizeof(T); }
  size_t size_bytes() const noexcept { return size_bytes_; }

  void copy_from_cuda_ptr(const CudaPtr<T> &other, const size_t n_bytes) {
    CUDA_CHECK(cudaMemcpy(ptr_, other.ptr_, n_bytes, cudaMemcpyDeviceToDevice));
  }

  void copy_from_host_ptr(const T *const p, const size_t n_bytes) {
    CUDA_CHECK(cudaMemcpy(ptr_, p, n_bytes, cudaMemcpyHostToDevice));
  }

private:
  T *ptr_;
  size_t size_bytes_;
};
