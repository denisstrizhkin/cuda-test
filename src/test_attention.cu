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

int main(void) {
  test_attention_cpu<float>();
  return EXIT_SUCCESS;
}
