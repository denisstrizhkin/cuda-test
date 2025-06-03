#pragma once

template <typename T> class CudaPtr {
public:
  explicit CudaPtr(const size_t n) noexcept
      : ptr_(nullptr), size_bytes_(n * sizeof(T)){
    if (n == 0) {
      return;
    }
    cudaMallocManaged(&ptr_, size_bytes_);
  }

  ~CudaPtr() noexcept {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
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

  CudaPtr &operator=(CudaPtr &&other) noexcept {
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

  T *get() const noexcept { return ptr_; }

  T &operator*() const noexcept { return *ptr_; }

  T *operator->() const noexcept { return ptr_; }

  T &operator[](const size_t i) noexcept { return ptr_[i]; }

  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  size_t size() const noexcept { return size_bytes_ / sizeof(T); }

  size_t size_bytes() const noexcept { return size_bytes_; }

  void copy_from_cuda_ptr(const CudaPtr<T>& other) noexcept {
    cudaMemcpy(ptr_, other.ptr_, other.size_bytes_, cudaMemcpyDeviceToDevice);
  }

private:
  T *ptr_;
  size_t size_bytes_;
};
