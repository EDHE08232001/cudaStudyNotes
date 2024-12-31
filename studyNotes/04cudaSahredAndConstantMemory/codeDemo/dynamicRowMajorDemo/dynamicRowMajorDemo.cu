#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BDIMX 32  // Number of threads in the x-dimension of a block
#define BDIMY 32  // Number of threads in the y-dimension of a block

/**
 * Kernel: setRowReadColDynamic
 * Description:
 * - Demonstrates dynamic shared memory allocation.
 * - Writes to shared memory in row-major order.
 * - Reads from shared memory in column-major order.
 *
 * Parameters:
 * - out: Pointer to the output array in global memory.
 */
__global__ void setRowReadColDynamic(int* out) {
    // Declare dynamically allocated shared memory (size determined at runtime)
    extern __shared__ int tile[];

    // Calculate row-major index for the thread
    int row_index = threadIdx.y * blockDim.x + threadIdx.x;

    // Calculate column-major index for the thread
    int col_index = threadIdx.x * blockDim.x + threadIdx.y;

    // Write to shared memory in row-major format
    tile[row_index] = row_index;

    // Synchronize all threads in the block to ensure the shared memory is updated
    __syncthreads();

    // Read from shared memory in column-major format and store the result in global memory
    out[row_index] = tile[col_index];
}

int main(int argc, char** argv) {
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

    // Launch the kernel with dynamically allocated shared memory
    int sharedMemBytes = matrixSize * sizeof(int);  // Size of shared memory in bytes
    setRowReadColDynamic<<<grid, block, sharedMemBytes>>>(d_out);

    // Copy the results back to the host
    cudaMemcpy(h_out, d_out, matrixBytes, cudaMemcpyDeviceToHost);

    // Print the resulting matrix
    printf("=== Output Matrix ===\n");
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
