#pragma once

#include "pcuda.cuh"
#include <utility>

template<typename T>
class Mat {
  public:
    explicit Mat(const size_t n, const size_t m) noexcept: n_(n), m_(m), data_(CudaPtr<T>(n * m)) {}
    ~Mat() noexcept = default;

    Mat(const Mat& other) noexcept: Mat(other.n_, other.m_) {
      data_.copy_from_cuda_ptr(other.data_);
    };

    Mat &operator=(const Mat& other) noexcept {
      if (this == &other) {
        return *this;
      }
      n_ = other.n_;
      m_ = other.m_;
      if (data_.size_bytes() < other.size_bytes()) {
        data_ = CudaPtr<T>(n_ * m_);
      }
      data_.copy_from_cuda_ptr(other.data_);
      return *this;
    }

    Mat(Mat &&other) noexcept: n_(other.n_), m_(other.m_), data_(std::move(other.data_)) {
      other.n_ = 0;
      other.m_ = 0;
    }

    Mat &operator=(Mat&& other) noexcept {
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

    const size_t rows() const noexcept { return n_; }
    const size_t cols() const noexcept { return m_; }
    const size_t size() const noexcept { return n_ * m_; }
    const size_t size_bytes() const noexcept { return data_.size_bytes(); }

    T& at(const size_t i, const size_t j) noexcept { return data_[i * m_ + j]; }
    const T& at(const size_t i, const size_t j) const noexcept { return data_[i * m_ + j]; }

  private:
    CudaPtr<T> data_;
    size_t n_;
    size_t m_;
};
