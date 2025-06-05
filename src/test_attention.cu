#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

#include "cpu.h"
#include "mat.cuh"

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
  for (auto i = 0; i < N; i++) {
    for (auto j = 0; j < D; j++) {
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

void test_gpu() {
  const size_t N = 4;
  const size_t D = 8;
  const auto mQ = cpu::random<float, N, D>();
  const auto mK = cpu::random<float, N, D>();
  const auto mV = cpu::random<float, N, D>();
  const auto result = cpu::attention<float, N, D>(mQ, mK, mV);
  assert(result.size() == N * D);
  const auto gMQ = Mat::Mat<float, N, D>(mQ);
  const auto gMK = Mat::Mat<float, N, D>(mK);
  const auto gMV = Mat::Mat<float, N, D>(mV);
  const auto gResult = Mat::Mat<float, N, D>::attention(gMQ, gMK, gMV);
  assert(gResult.size() == N * D);
  for (auto i = 0; i < N; i++) {
    for (auto j = 0; j < D; j++) {
      const auto got = gResult.at(i ,j);
      const auto expected = result[i * D + j];
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
  test_gpu();
  return EXIT_SUCCESS;
}
