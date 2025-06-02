#include <cstdio>
#include "util.cuh" // Include shared utilities

__global__ void vector_sum_kernel(float* sum_out, const float* vec, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;

    // Load data into shared memory
    if (i < n) sdata[tid] = vec[i];
    if (i + blockDim.x < n) sdata[tid + blockDim.x] = vec[i + blockDim.x];
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (i + s) < n) { // Check bounds for actual data
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block sum to global memory
    if (tid == 0) {
        atomicAdd(sum_out, sdata[0]);
    }
}

int main() {
    int n = 1000;
    float *vec;
    float h_sum = 0.0f;
    size_t size = n * sizeof(float);

    // Allocate host memory
    vec = (float*)malloc(size);

    // Initialize host data
    for (int i = 0; i < n; ++i) {
        vec[i] = (float)i;
    }

    // Allocate device memory
    float *d_vec, *d_sum;
    cudaMalloc(&d_vec, size);
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float)); // Initialize d_sum to 0

    // Copy host data to device
    cudaMemcpy(d_vec, vec, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + (blockSize * 2) - 1) / (blockSize * 2); // Adjust for 2 loads per thread
    vector_sum_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_sum, d_vec, n);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Vector Sum Result: %.2f\n", h_sum);

    // Use shared utility function
    print_vector_info(vec, n);

    // Free memory
    free(vec);
    cudaFree(d_vec); cudaFree(d_sum);

    return 0;
}
