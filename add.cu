#include <cmath>
#include <iostream>

template <typename T> class CudaPtr {
public:
  explicit CudaPtr(const size_t n)
      : ptr_(nullptr), size_bytes_(n * sizeof(T)) {
    if (n == 0) {
      return;
    }
    cudaMallocManaged(&ptr_, size_bytes_);
  }

  ~CudaPtr() {
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

  T *get() const { return ptr_; }

  T &operator*() const { return *ptr_; }

  T *operator->() const { return ptr_; }

  T &operator[](const size_t i) { return ptr_[i]; }

  explicit operator bool() const { return ptr_ != nullptr; }

  size_t size() const { return size_bytes_; }

private:
  T *ptr_;
  size_t size_bytes_;
};

// function to add the elements of two arrays
__global__ void add(const size_t n, float *x, float *y) {
  const auto i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < n) {
    y[i] = x[i] + y[i];
  }
}

void print_gpu_info() {
  int deviceId;
  cudaGetDevice(&deviceId); // Get current device ID
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceId);

  std::cout << "Device: " << deviceProp.name << "\n";
  std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << "\n";
  std::cout << "Max grid size (x): " << deviceProp.maxGridSize[0] << "\n";
  std::cout << "Number of SMs: " << deviceProp.multiProcessorCount << "\n";
}

int main(void) {
  print_gpu_info();
  
  const auto N = 200'000'000; // 1M elements

  auto x = CudaPtr<float>(N);
  auto y = CudaPtr<float>(N);

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  const auto blockSize = 1024;
  const auto numBlocks = (N + blockSize - 1) / blockSize;
  // Run kernel on 1M elements on the CPU
  cudaMemPrefetchAsync(x.get(), x.size(), 0, 0);
  cudaMemPrefetchAsync(x.get(), x.size(), 0, 0);
  add<<<numBlocks, blockSize>>>(N, x.get(), y.get());
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  return 0;
}
