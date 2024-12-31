#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BDIMX 64 // Threads per block in x-dimension
#define BDIMY 8  // Threads per block in y-dimension
#define IPAD 2   // Padding to avoid shared memory bank conflicts

/**
 * Kernel: transpose_read_raw_write_column_benchmark
 * - Reads the input matrix row-by-row and writes the transposed matrix column-by-column.
 *
 * Parameters:
 * - mat: Input matrix stored in row-major order.
 * - transpose: Output transposed matrix.
 * - nx: Number of columns in the input matrix.
 * - ny: Number of rows in the input matrix.
 */
__global__ void transpose_read_raw_write_column_benchmark(int* mat, int* transpose, int nx, int ny) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x; // Global column index
    int iy = blockDim.y * blockIdx.y + threadIdx.y; // Global row index

    if (ix < nx && iy < ny) {
        // Read from input matrix and write to transposed matrix
        transpose[ix * ny + iy] = mat[iy * nx + ix];
    }
}

/**
 * Kernel: transpose_smem
 * - Uses shared memory to optimize matrix transpose.
 * - Stores the input matrix in shared memory in row-major order, then writes the transposed result.
 *
 * Parameters:
 * - in: Input matrix stored in row-major order.
 * - out: Output transposed matrix.
 * - nx: Number of columns in the input matrix.
 * - ny: Number of rows in the input matrix.
 */
__global__ void transpose_smem(int* in, int* out, int nx, int ny) {
    __shared__ int tile[BDIMY][BDIMX]; // Shared memory tile

    int ix = blockDim.x * blockIdx.x + threadIdx.x; // Global column index
    int iy = blockDim.y * blockIdx.y + threadIdx.y; // Global row index

    int in_index = iy * nx + ix; // Linear index for input matrix

    int thread_1d_idx = threadIdx.y * blockDim.x + threadIdx.x; // Linear thread index
    int i_row = thread_1d_idx / blockDim.y; // Shared memory row index
    int i_col = thread_1d_idx % blockDim.y; // Shared memory column index

    int out_ix = blockIdx.y * blockDim.y + i_col; // Transposed column index
    int out_iy = blockIdx.x * blockDim.x + i_row; // Transposed row index
    int out_index = out_iy * ny + out_ix; // Linear index for output matrix

    if (ix < nx && iy < ny) {
        // Load input matrix to shared memory
        tile[threadIdx.y][threadIdx.x] = in[in_index];

        // Wait for all threads to finish loading
        __syncthreads();

        // Write transposed data from shared memory to output matrix
        out[out_index] = tile[i_col][i_row];
    }
}

/**
 * Kernel: transpose_smem_pad
 * - Similar to transpose_smem but includes padding to avoid shared memory bank conflicts.
 *
 * Parameters:
 * - in: Input matrix stored in row-major order.
 * - out: Output transposed matrix.
 * - nx: Number of columns in the input matrix.
 * - ny: Number of rows in the input matrix.
 */
__global__ void transpose_smem_pad(int* in, int* out, int nx, int ny) {
    __shared__ int tile[BDIMY][BDIMX + IPAD]; // Shared memory tile with padding

    int ix = blockDim.x * blockIdx.x + threadIdx.x; // Global column index
    int iy = blockDim.y * blockIdx.y + threadIdx.y; // Global row index

    int in_index = iy * nx + ix; // Linear index for input matrix

    int thread_1d_idx = threadIdx.y * blockDim.x + threadIdx.x; // Linear thread index
    int i_row = thread_1d_idx / blockDim.y; // Shared memory row index
    int i_col = thread_1d_idx % blockDim.y; // Shared memory column index

    int out_ix = blockIdx.y * blockDim.y + i_col; // Transposed column index
    int out_iy = blockIdx.x * blockDim.x + i_row; // Transposed row index
    int out_index = out_iy * ny + out_ix; // Linear index for output matrix

    if (ix < nx && iy < ny) {
        // Load input matrix to shared memory with padding
        tile[threadIdx.y][threadIdx.x] = in[in_index];

        // Wait for all threads to finish loading
        __syncthreads();

        // Write transposed data from shared memory to output matrix
        out[out_index] = tile[i_col][i_row];
    }
}

/**
 * Kernel: transpose_smem_pad_unrolling
 * - Unrolls loops to improve performance by loading and storing two elements per thread.
 * - Uses shared memory with padding to avoid bank conflicts.
 *
 * Parameters:
 * - in: Input matrix stored in row-major order.
 * - out: Output transposed matrix.
 * - nx: Number of columns in the input matrix.
 * - ny: Number of rows in the input matrix.
 */
__global__ void transpose_smem_pad_unrolling(int* in, int* out, int nx, int ny) {
    __shared__ int tile[BDIMY * (2 * BDIMX + IPAD)]; // Shared memory tile with padding

    int ix = 2 * blockDim.x * blockIdx.x + threadIdx.x; // Global column index (unrolled)
    int iy = blockDim.y * blockIdx.y + threadIdx.y;    // Global row index

    int in_index = iy * nx + ix; // Linear index for input matrix

    int thread_1d_idx = threadIdx.y * blockDim.x + threadIdx.x; // Linear thread index
    int i_row = thread_1d_idx / blockDim.y; // Shared memory row index
    int i_col = thread_1d_idx % blockDim.y; // Shared memory column index

    int out_ix = blockIdx.y * blockDim.y + i_col; // Transposed column index
    int out_iy = 2 * blockIdx.x * blockDim.x + i_row; // Transposed row index (unrolled)
    int out_index = out_iy * ny + out_ix; // Linear index for output matrix

    if (ix < nx && iy < ny) {
        int row_idx = threadIdx.y * (2 * blockDim.x + IPAD) + threadIdx.x; // Shared memory row index

        // Load input matrix to shared memory (unrolled)
        tile[row_idx] = in[in_index];
        tile[row_idx + BDIMX] = in[in_index + BDIMX];

        // Wait for all threads to finish loading
        __syncthreads();

        int col_idx = i_col * (2 * blockDim.x + IPAD) + i_row; // Shared memory column index

        // Write transposed data from shared memory to output matrix (unrolled)
        out[out_index] = tile[col_idx];
        out[out_index + ny * BDIMX] = tile[col_idx + BDIMX];
    }
}

/**
 * Main function
 * - Initializes matrices, launches the shared memory kernel, and verifies the result.
 */
int main() {
    const int nx = 1024; // Number of columns
    const int ny = 1024; // Number of rows
    const int matrix_size = nx * ny;
    const int matrix_bytes = matrix_size * sizeof(int);

    // Host memory allocation
    int* h_in = (int*)malloc(matrix_bytes);
    int* h_out = (int*)malloc(matrix_bytes);

    // Initialize input matrix with values
    for (int i = 0; i < matrix_size; ++i) {
        h_in[i] = i;
    }

    // Device memory allocation
    int* d_in;
    int* d_out;
    cudaMalloc(&d_in, matrix_bytes);
    cudaMalloc(&d_out, matrix_bytes);

    // Copy input data to device
    cudaMemcpy(d_in, h_in, matrix_bytes, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block(BDIMX, BDIMY);
    dim3 grid((nx + BDIMX - 1) / BDIMX, (ny + BDIMY - 1) / BDIMY);

    // Launch shared memory kernel
    transpose_smem<<<grid, block>>>(d_in, d_out, nx, ny);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_out, d_out, matrix_bytes, cudaMemcpyDeviceToHost);

    // Verify result
    bool success = true;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            if (h_out[j * nx + i] != h_in[i * ny + j]) {
                success = false;
                printf("Mismatch at (%d, %d): %d != %d\n", i, j, h_out[j * nx + i], h_in[i * ny + j]);
            }
        }
    }

    if (success) {
        printf("Transpose successful!\n");
    } else {
        printf("Transpose failed!\n");
    }

    // Free memory
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}