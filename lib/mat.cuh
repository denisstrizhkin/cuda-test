#pragma once

#include "pcuda.cuh"

template <typename T>
__global__ void mat_mul(const T *const __restrict__ mA,
                        const T *const __restrict__ mB,
                        T *const __restrict__ mC,
                        const size_t A_rows,
                        const size_t A_cols_B_rows,
                        const size_t B_cols);

template <typename T>
class Mat {
public:
    explicit Mat(const size_t n, const size_t m) noexcept;
    ~Mat() noexcept = default;

    Mat(const Mat &other) noexcept;

    Mat &operator=(const Mat &other) noexcept;

    Mat(Mat &&other) noexcept;

    Mat &operator=(Mat &&other) noexcept;

    const size_t rows() const noexcept { return n_; }
    const size_t cols() const noexcept { return m_; }
    const size_t size() const noexcept { return n_ * m_; }
    const size_t size_bytes() const noexcept { return data_.size_bytes(); }

    T &at(const size_t i, const size_t j) noexcept;
    const T &at(const size_t i, const size_t j) const noexcept;

    Mat dot(const Mat &other) const noexcept;

    static Mat random(const size_t n, const size_t m) noexcept;

private:
    CudaPtr<T> data_;
    size_t n_;
    size_t m_;
};;
