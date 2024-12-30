#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define INIT_ONE_TO_TEN 1

// Kernel to transpose matrix with unrolling: Read by column and write by row
__global__ void transpose_unrolling_read_column_write_row(int* mat, int* transpose, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x * 4 + threadIdx.x; // Calculate global id for x with unrolling
    int iy = blockIdx.y * blockDim.y + threadIdx.y;     // Calculate global id for y

    int ti = iy * nx + ix; // Input matrix index
    int to = ix * ny + iy; // Transposed matrix index

    // Ensure that the indices are within bounds and perform unrolled memory access
    if (ix + 3 * blockDim.x < nx && iy < ny) {
        transpose[to] = mat[ti];
        transpose[to + ny * blockDim.x] = mat[ti + blockDim.x];
        transpose[to + ny * 2 * blockDim.x] = mat[ti + 2 * blockDim.x];
        transpose[to + ny * 3 * blockDim.x] = mat[ti + 3 * blockDim.x];
    }
}

// Function to initialize the matrix with integers between 1 and 10
void initialize(int* mat, int size, int mode) {
    if (mode == INIT_ONE_TO_TEN) {
        for (int i = 0; i < size; i++) {
            mat[i] = (i % 10) + 1; // Fill with numbers 1 to 10 cyclically
        }
    }
}

int main(int argc, char** argv) {
    // Default matrix and block dimensions
    int nx = 1024; // Number of columns in the matrix
    int ny = 1024; // Number of rows in the matrix
    int block_x = 128; // Number of threads per block in x-dimension
    int block_y = 8;   // Number of threads per block in y-dimension

    int size = nx * ny; // Total number of elements in the matrix
    int byte_size = sizeof(int) * size; // Total size in bytes

    printf("Matrix transpose with unrolling for %d by %d matrix with block size %d by %d \n", nx, ny, block_x, block_y);

    // Allocate host memory
    int* h_mat_array = (int*)malloc(byte_size); // Host memory for the original matrix
    int* h_trans_array_unroll = (int*)malloc(byte_size); // Host memory for the transposed matrix

    // Initialize the matrix
    initialize(h_mat_array, size, INIT_ONE_TO_TEN);

    // Allocate device memory
    int* d_mat_array;
    int* d_trans_array;
    cudaMalloc((void**)&d_mat_array, byte_size); // Device memory for the original matrix
    cudaMalloc((void**)&d_trans_array, byte_size); // Device memory for the transposed matrix

    // Copy data from host to device
    cudaMemcpy(d_mat_array, h_mat_array, byte_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blocks(block_x, block_y); // Block dimensions
    dim3 grid((nx + block_x * 4 - 1) / (block_x * 4), (ny + block_y - 1) / block_y); // Grid dimensions with unrolling

    // Launch the kernel with unrolling
    transpose_unrolling_read_column_write_row<<<grid, blocks>>>(d_mat_array, d_trans_array, nx, ny);

    // Synchronize to ensure kernel execution is complete
    cudaDeviceSynchronize();

    // Copy the transposed matrix back to the host
    cudaMemcpy(h_trans_array_unroll, d_trans_array, byte_size, cudaMemcpyDeviceToHost);

    // Print a portion of the original and transposed matrices for verification
    printf("Original Matrix (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_mat_array[i]);
    }
    printf("\n\nTransposed Matrix with Unrolling (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_trans_array_unroll[i * nx]); // Access column-major order
    }
    printf("\n");

    // Free device memory
    cudaFree(d_mat_array);
    cudaFree(d_trans_array);

    // Free host memory
    free(h_mat_array);
    free(h_trans_array_unroll);

    // Reset the device
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
