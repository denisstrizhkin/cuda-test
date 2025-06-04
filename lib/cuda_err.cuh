#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

class CudaException : public std::runtime_error {
public:
  explicit CudaException(const std::string &msg) : std::runtime_error(msg) {}
};

void cudaCheckError(const cudaError_t code, const char *const file,
                    const int line) {
  if (code == cudaSuccess) {
    return;
  }
  std::stringstream oss;
  oss << "CUDA Error at " << file << ":" << line << ": "
      << cudaGetErrorString(code);
  const auto msg = oss.str();
  std::cerr << msg << "\n";
  throw CudaException(msg);
}

#define CUDA_CHECK(code)                                                       \
  {                                                                            \
    cudaCheckError((code), __FILE__, __LINE__);                                \
  }
