#pragma once

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "rand.h"

namespace cpu {
template <typename T, size_t N, size_t M>
void check_dims(const std::vector<T> &v) {
  if (v.size() != N * M) {
    std::stringstream oss;
    oss << "vector size does not match " << N << "x" << M << " dims";
    throw std::invalid_argument(oss.str());
  }
}

template <typename T, size_t N> std::vector<T> random() {
  std::vector<T> v(N);
  for (auto i = 0; i < N; i++) {
    v[i] = Random::getInstance().next();
  }
  return v;
}

template <typename T, size_t N, size_t M> inline std::vector<T> random() {
  return random<T, N * M>();
}

template <typename T, size_t N, size_t M, size_t K>
std::vector<T> dot(const std::vector<T> &a, const std::vector<T> &b) {
  check_dims<T, N, M>(a);
  check_dims<T, M, K>(b);
  std::vector<T> c(N * K);
  if (N == 0 || M == 0 || K == 0) {
    return c;
  }
  for (auto i = 0; i < N; i++) {
    for (auto j = 0; j < K; j++) {
      T sum = 0;
      for (auto k = 0; k < M; k++) {
        sum += a[i * M + k] * b[k * K + j];
      }
      c[i * K + j] = sum;
    }
  }
  return c;
}

template <typename T, size_t N, size_t M>
std::vector<T> transpose(const std::vector<T> &v) {
  check_dims<T, N, M>(v);
  std::vector<T> t(N * M);
  for (auto i = 0; i < N; ++i) {
    for (auto j = 0; j < M; ++j) {
      t[j * N + i] = v[i * M + j];
    }
  }
  return t;
}

template <typename T, size_t N, size_t M>
std::vector<T> softmax(const std::vector<T> &v) {
  check_dims<T, N, M>(v);
  std::vector<T> p(N * M);
  for (auto i = 0; i < N; ++i) {
    const auto max_val =
        *std::max_element(v.begin() + i * M, v.begin() + (i + 1) * M);
    T exp_sum = 0;
    for (auto j = 0; j < M; ++j) {
      const auto ed = std::exp(v[i * M + j] - max_val);
      p[i * M + j] = ed;
      exp_sum += ed;
    }
    for (auto j = 0; j < M; ++j) {
      p[i * M + j] /= exp_sum;
    }
  }
  return p;
}

template <typename T, size_t N, size_t D>
std::vector<T> attention(const std::vector<T> &mQ, const std::vector<T> &mK,
                         const std::vector<T> &mV) {
  check_dims<T, N, D>(mQ);
  check_dims<T, N, D>(mK);
  check_dims<T, N, D>(mV);
  const auto mKT = transpose<T, N, D>(mK);
  const auto s = dot<T, N, D, N>(mQ, mKT);
  const auto p = softmax<T, N, N>(s);
  return dot<T, N, N, D>(p, mV);
}

}; // namespace cpu
