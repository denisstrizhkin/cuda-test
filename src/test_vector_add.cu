#include <cstdio>
#include "util.cuh" // Include shared utilities

__global__ void vector_add_kernel(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 10;
    float *a, *b, *out;
    size_t size = n * sizeof(float);

    // Allocate host memory
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    out = (float*)malloc(size);

    // Initialize host data
    for (int i = 0; i < n; ++i) {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_out, size);

    // Copy host data to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vector_add_kernel<<<numBlocks, blockSize>>>(d_out, d_a, d_b, n);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    printf("Vector Add Result (first few elements):\n");
    for (int i = 0; i < 5; ++i) {
        printf("%.2f + %.2f = %.2f\n", a[i], b[i], out[i]);
    }

    // Use shared utility function
    print_vector_info(out, n);

    // Free memory
    free(a); free(b); free(out);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);

    return 0;
}

