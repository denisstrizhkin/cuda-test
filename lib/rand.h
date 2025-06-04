#pragma once

#include <random>

class Random {
private:
  Random() {
    std::random_device rd;
    engine_ = std::mt19937(rd());
    distribution_ = std::uniform_real_distribution<double>(0.0, 1.0);
  }

  Random(const Random &) = delete;
  Random &operator=(const Random &) = delete;
  Random(Random &&) = delete;
  Random &operator=(Random &&) = delete;

public:
  static Random &getInstance() noexcept {
    static Random instance;
    return instance;
  }

  double next() { return distribution_(engine_); }

private:
  std::mt19937 engine_;
  std::uniform_real_distribution<double> distribution_;
};
