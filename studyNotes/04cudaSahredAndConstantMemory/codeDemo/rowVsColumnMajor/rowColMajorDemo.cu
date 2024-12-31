#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BDIMX 32
#define BDIMY 32

/**
 * Accessing Shared Memory in Row-Major Format
 */
__global__ void setRowReadCol(int* out) {
    __shared__ int tile[BDIMY][BDIMX];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // Write to shared memory in row-major format
    tile[threadIdx.y][threadIdx.x] = idx;

    // Synchronize all threads in the block
    __syncthreads();

    // Read from shared memory in column-major format
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

/**
 * Accessing Shared Memory in Column-Major Format
 */
__global__ void setColReadRow(int* out) {
    __shared__ int tile[BDIMY][BDIMX];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // Write to shared memory in column-major format
    tile[threadIdx.x][threadIdx.y] = idx;

    // Synchronize all threads in the block
    __syncthreads();

    // Read from shared memory in row-major format
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

/**
 * Accessing Shared Memory in Row-Major Format
 */
__global__ void setRowReadRow(int* out) {
    __shared__ int tile[BDIMY][BDIMX];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // Write to shared memory in row-major format
    tile[threadIdx.y][threadIdx.x] = idx;

    // Synchronize all threads in the block
    __syncthreads();

    // Read from shared memory in row-major format
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

void printMatrix(int* mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%4d ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    const int matrixSize = BDIMX * BDIMY;
    const int matrixBytes = matrixSize * sizeof(int);

    // Allocate memory on host
    int* h_out = (int*)malloc(matrixBytes);

    // Allocate memory on device
    int* d_out;
    cudaMalloc(&d_out, matrixBytes);

    // Define block and grid dimensions
    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);

    // Launch kernels and display results
    printf("=== Kernel: setRowReadCol ===\n");
    setRowReadCol<<<grid, block>>>(d_out);
    cudaMemcpy(h_out, d_out, matrixBytes, cudaMemcpyDeviceToHost);
    printMatrix(h_out, BDIMY, BDIMX);

    printf("\n=== Kernel: setColReadRow ===\n");
    setColReadRow<<<grid, block>>>(d_out);
    cudaMemcpy(h_out, d_out, matrixBytes, cudaMemcpyDeviceToHost);
    printMatrix(h_out, BDIMY, BDIMX);

    printf("\n=== Kernel: setRowReadRow ===\n");
    setRowReadRow<<<grid, block>>>(d_out);
    cudaMemcpy(h_out, d_out, matrixBytes, cudaMemcpyDeviceToHost);
    printMatrix(h_out, BDIMY, BDIMX);

    // Free memory
    free(h_out);
    cudaFree(d_out);

    return 0;
}