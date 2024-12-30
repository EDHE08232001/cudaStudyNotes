#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define INIT_ONE_TO_TEN 1

// Kernel to transpose matrix: Read by row and write by column
__global__ void transpose_read_row_write_column(int* mat, int* transpose, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // global id for x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // global id for y

    if (ix < nx && iy < ny) {
        transpose[ix * ny + iy] = mat[iy * nx + ix];
    }
}

// Kernel to transpose matrix: Read by column and write by row
__global__ void transpose_read_column_write_row(int* mat, int* transpose, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // global id for x
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // global id for y

    if (ix < nx && iy < ny) {
        transpose[iy * nx + ix] = mat[ix * ny + iy];
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
    int nx = 1024;
    int ny = 1024;
    int block_x = 128;
    int block_y = 8;

    int size = nx * ny;
    int byte_size = sizeof(int) * size;

    printf("Matrix transpose for %d by %d matrix with block size %d by %d \n", nx, ny, block_x, block_y);

    // Memory usage before allocation
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("Before allocation - Free memory: %lu bytes, Total memory: %lu bytes\n", free_mem, total_mem);

    // Allocate host memory
    int* h_mat_array = (int*)malloc(byte_size);
    int* h_trans_array_row = (int*)malloc(byte_size);
    int* h_trans_array_col = (int*)malloc(byte_size);

    // Initialize the matrix
    initialize(h_mat_array, size, INIT_ONE_TO_TEN);

    // Allocate device memory
    int* d_mat_array;
    int* d_trans_array;
    cudaMalloc((void**)&d_mat_array, byte_size);
    cudaMalloc((void**)&d_trans_array, byte_size);

    // Copy data from host to device
    cudaMemcpy(d_mat_array, h_mat_array, byte_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blocks(block_x, block_y);
    dim3 grid((nx + block_x - 1) / block_x, (ny + block_y - 1) / block_y);

    // Measure kernel execution time
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing kernel 1
    cudaEventRecord(start, 0);
    transpose_read_row_write_column<<<grid, blocks>>>(d_mat_array, d_trans_array, nx, ny);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel 1 execution time (Read Row, Write Column): %.3f ms\n", elapsedTime);

    // Copy the transposed matrix back to the host
    cudaMemcpy(h_trans_array_row, d_trans_array, byte_size, cudaMemcpyDeviceToHost);

    // Start timing kernel 2
    cudaEventRecord(start, 0);
    transpose_read_column_write_row<<<grid, blocks>>>(d_mat_array, d_trans_array, nx, ny);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel 2 execution time (Read Column, Write Row): %.3f ms\n", elapsedTime);

    // Copy the second transposed matrix back to the host
    cudaMemcpy(h_trans_array_col, d_trans_array, byte_size, cudaMemcpyDeviceToHost);

    // Memory usage after execution
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("After execution - Free memory: %lu bytes, Total memory: %lu bytes\n", free_mem, total_mem);

    // Print a portion of the original and transposed matrices for verification
    printf("Original Matrix (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_mat_array[i]);
    }
    printf("\n\nTransposed Matrix (Read Row, Write Column, first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_trans_array_row[i * nx]);
    }
    printf("\n\nTransposed Matrix (Read Column, Write Row, first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_trans_array_col[i * nx]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_mat_array);
    cudaFree(d_trans_array);

    // Free host memory
    free(h_mat_array);
    free(h_trans_array_row);
    free(h_trans_array_col);

    // Reset the device
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
