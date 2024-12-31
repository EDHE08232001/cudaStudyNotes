#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BDIMX 32  // Number of threads in x-dimension of a block
#define BDIMY 32  // Number of threads in y-dimension of a block
#define IPAD 1    // Padding to avoid shared memory bank conflicts

/**
 * Kernel: setRowReadColPad
 * Description:
 * - Demonstrates the use of statically allocated shared memory.
 * - Adds padding to shared memory to avoid bank conflicts.
 * - Writes in row-major order and reads in column-major order.
 *
 * Parameters:
 * - out: Pointer to the output array in global memory.
 */
__global__ void setRowReadColPad(int* out) {
    // Declare statically allocated shared memory with padding
    __shared__ int tile[BDIMY][BDIMX + IPAD];

    // Calculate the global index for the thread
    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // Write to shared memory in row-major format
    tile[threadIdx.y][threadIdx.x] = idx;

    // Synchronize all threads to ensure shared memory is updated
    __syncthreads();

    // Read from shared memory in column-major format and store the result in global memory
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

/**
 * Kernel: setRowReadColDynamicPad
 * Description:
 * - Demonstrates the use of dynamically allocated shared memory.
 * - Adds padding dynamically to avoid bank conflicts.
 * - Writes in row-major order and reads in column-major order.
 *
 * Parameters:
 * - out: Pointer to the output array in global memory.
 */
__global__ void setRowReadColDynamicPad(int* out) {
    // Declare dynamically allocated shared memory
    extern __shared__ int tile[];

    // Calculate the row-major index with padding
    int row_index = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;

    // Calculate the column-major index with padding
    int col_index = threadIdx.x * (blockDim.x + IPAD) + threadIdx.y;

    // Write to shared memory in row-major format
    tile[row_index] = row_index;

    // Synchronize all threads to ensure shared memory is updated
    __syncthreads();

    // Read from shared memory in column-major format and store the result in global memory
    out[row_index] = tile[col_index];
}

/**
 * Main function
 * - Initializes data and launches the CUDA kernels.
 * - Copies results back to host memory and prints the output.
 */
int main() {
    const int matrixSize = BDIMX * BDIMY;         // Total number of elements
    const int matrixBytes = matrixSize * sizeof(int);  // Total size in bytes

    // Allocate memory on the host
    int* h_out = (int*)malloc(matrixBytes);

    // Allocate memory on the device
    int* d_out;
    cudaMalloc(&d_out, matrixBytes);

    // Define block and grid dimensions
    dim3 block(BDIMX, BDIMY);  // Single block with 32x32 threads
    dim3 grid(1, 1);          // Single grid

    // Launch the kernel with statically padded shared memory
    printf("=== Kernel: setRowReadColPad ===\n");
    setRowReadColPad<<<grid, block>>>(d_out);
    cudaMemcpy(h_out, d_out, matrixBytes, cudaMemcpyDeviceToHost);

    // Print the resulting matrix
    for (int i = 0; i < BDIMY; ++i) {
        for (int j = 0; j < BDIMX; ++j) {
            printf("%4d ", h_out[i * BDIMX + j]);
        }
        printf("\n");
    }

    // Launch the kernel with dynamically padded shared memory
    printf("\n=== Kernel: setRowReadColDynamicPad ===\n");
    int sharedMemBytes = matrixSize * sizeof(int);  // Size of dynamic shared memory
    setRowReadColDynamicPad<<<grid, block, sharedMemBytes>>>(d_out);
    cudaMemcpy(h_out, d_out, matrixBytes, cudaMemcpyDeviceToHost);

    // Print the resulting matrix
    for (int i = 0; i < BDIMY; ++i) {
        for (int j = 0; j < BDIMX; ++j) {
            printf("%4d ", h_out[i * BDIMX + j]);
        }
        printf("\n");
    }

    // Free allocated memory
    free(h_out);
    cudaFree(d_out);

    return 0;
}
