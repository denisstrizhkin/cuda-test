#include <cstdio>
#include "util.cuh"

// Example function definition
void print_vector_info(const float* vec, int size) {
    if (size > 0) {
        printf("Vector size: %d, First element: %.2f, Last element: %.2f\n", size, vec[0], vec[size - 1]);
    } else {
        printf("Empty vector.\n");
    }
}
