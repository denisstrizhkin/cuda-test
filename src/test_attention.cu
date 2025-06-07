#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cpu.h"
#include "mat.cuh"

template <typename T, size_t N, size_t M>
void compare(const Mat::Mat<T, N, M> &m, const std::vector<T> &v) {
  const size_t size = N * M;
  const size_t v_size = v.size();
  if (v_size != size) {
    std::cout << "vector size: expected: " << size << ", got: " << v_size
              << "\n";
    assert(false);
  }
  const size_t m_size = m.size();
  if (m_size != size) {
    std::cout << "mat size: expected: " << size << ", got: " << m_size << "\n";
    assert(false);
  }
  const size_t m_rows = m.rows();
  if (m_rows != N) {
    std::cout << "mat rows: expected: " << N << ", got: " << m_rows << "\n";
    assert(false);
  }
  const size_t m_cols = m.cols();
  if (m_cols != M) {
    std::cout << "mat cols: expected: " << M << ", got: " << m_cols << "\n";
    assert(false);
  }
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < M; j++) {
      const T got = m.at(i, j);
      const T expected = v[i * M + j];
      const T diff = std::abs(expected - got);
      if (diff > 1e-6) {
        std::cout << "at (" << i << "," << j << ") expected: " << expected
                  << ", got: " << got << "\n";
        assert(false);
      }
    }
  }
}

template <typename T, size_t N, size_t M> void test_from_vector() {
  const auto v = cpu::random<T, N, M>();
  const auto m = Mat::Mat<T, N, M>(v);
  compare(m, v);
}

template <typename T, size_t N, size_t M> void test_transpose() {
  const auto a = cpu::random<T, N, M>();
  const auto aT = cpu::transpose<T, N, M>(a);
  const auto mA = Mat::Mat<T, N, M>(a);
  const auto mAT = mA.transpose();
  compare(mAT, aT);
}

template <typename T, size_t N, size_t M> void test_softmax() {
  const auto a = cpu::random<T, N, M>();
  const auto smaxA = cpu::softmax<T, N, M>(a);
  const auto mA = Mat::Mat<T, N, M>(a);
  const auto smaxMA = mA.softmax();
  compare(smaxMA, smaxA);
}

template <typename T, size_t N, size_t M> void test_attention() {
  const auto q = cpu::random<T, N, M>();
  const auto k = cpu::random<T, N, M>();
  const auto v = cpu::random<T, N, M>();
  const auto o = cpu::attention<T, N, M>(q, k, v);
  const auto mQ = Mat::Mat<T, N, M>(q);
  const auto mK = Mat::Mat<T, N, M>(k);
  const auto mV = Mat::Mat<T, N, M>(v);
  const auto mO = Mat::Mat<T, N, M>::attention(mQ, mK, mV);
  compare(mO, o);
}

template <typename T> void test_attention_cpu() {
  const size_t N = 3;
  const size_t D = 2;
  const std::vector<T> q = {
      1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
  };
  const std::vector<T> k = {
      1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
  };
  const std::vector<T> v = {
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
  };
  const auto o = cpu::attention<T, N, D>(q, k, v);
  const std::vector<T> o_expected = {
      2.27158774, 3.27159260, 2.99901458, 4.00001944, 2.46607088, 3.46607088,
  };
  assert(o.size() == N * D);
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < D; j++) {
      const auto k = i * D + j;
      const auto got = o[k];
      const auto expected = o_expected[k];
      const auto diff = std::abs(expected - got);
      if (diff > 1e-3) {
        std::cout << "at (" << i << "," << j << ") expected: " << expected
                  << ", got: " << got << "\n";
        assert(false);
      }
    }
  }
}

int main(void) {
  test_attention_cpu<float>();
  const size_t N = 8;
  const size_t D = 4;
  test_from_vector<float, N, D>();
  test_transpose<float, N, D>();
  test_softmax<float, N, D>();
  test_attention<float, N, N>();
  test_from_vector<float, D, N>();
  test_transpose<float, D, N>();
  test_softmax<float, D, N>();
  test_attention<float, D, N>();
  return EXIT_SUCCESS;
}
