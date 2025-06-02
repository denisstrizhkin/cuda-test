#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

class RandomDoubleProvider {
private:
  RandomDoubleProvider() {
    std::random_device rd;
    engine_ = std::mt19937(rd());
    distribution_ = std::uniform_real_distribution<double>(0.0, 1.0);
  }

  RandomDoubleProvider(const RandomDoubleProvider &) = delete;
  RandomDoubleProvider &operator=(const RandomDoubleProvider &) = delete;
  RandomDoubleProvider(RandomDoubleProvider &&) =
      delete; // Delete move constructor
  RandomDoubleProvider &
  operator=(RandomDoubleProvider &&) = delete; // Delete move assignment

public:
  static RandomDoubleProvider &getInstance() {
    static RandomDoubleProvider instance;
    return instance;
  }

  double getRandomDouble() { return distribution_(engine_); }

  double getRandomDouble(double min, double max) {
    if (min > max) {
      std::swap(min, max); // Ensure min is less than or equal to max
    }
    std::uniform_real_distribution<double> custom_dist(min, max);
    return custom_dist(engine_);
  }

private:
  std::mt19937 engine_; // The random number engine
  std::uniform_real_distribution<double> distribution_; // The distribution
};

std::vector<double> get_random_vector(const int n) {
  std::vector<double> v(n);
  auto &random_provider = RandomDoubleProvider::getInstance();
  for (double &e : v) {
    e = random_provider.getRandomDouble();
  }
  return v;
}

template <typename T>
std::vector<T> add(const T *p_x, const T *p_y, const size_t n) {
  if (p_x == nullptr || p_y == nullptr) {
    throw std::invalid_argument("Input arrays cannot be null.");
  }
  std::vector<T> r(n);
  std::transform(p_x, p_x + n, p_y, r.data(), std::plus<T>());
  return r;
}

int main(void) {
  static const size_t N = 1 << 20; // 1M elements

  auto x = get_random_vector(N);
  auto y = get_random_vector(N);

  auto r = add(x.data(), y.data(), N);

  for (auto i = 0; i < N; i++) {
    const auto a = x[i] + y[i];
    const auto diff = std::abs(a - r[i]);
    if (diff > 1e-12) {
      std::cout << "Erm... " << a << " != " << r[i] << "\n";
      return 1;
    }
  }
  std::cout << "All good!\n";
  return 0;
}
