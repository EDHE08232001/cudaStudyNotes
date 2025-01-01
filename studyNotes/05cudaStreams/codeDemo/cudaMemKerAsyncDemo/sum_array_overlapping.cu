#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Function to initialize an array with random values between 1 and 10
void initialize(int *array, int size) {
    srand(time(NULL)); // Seed the random number generator
    for (int i = 0; i < size; i++) {
        array[i] = (rand() % 10) + 1; // Random values in the range [1, 10]
    }
}

// Function to sum arrays on the host (CPU)
void sumArraysOnHostx(int *A, int *B, int *C, const int N) {
    for (int idx = 0; idx < N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

// CUDA kernel to sum arrays on the device (GPU)
__global__ void sum_array_overlap(int *a, int *b, int *c, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        c[gid] = a[gid] + b[gid];
    }
}

int main() {
    // Define problem size
    int size = 1 << 25; // Number of elements in the arrays
    int block_size = 128;

    // Total bytes needed to hold all elements
    size_t NO_BYTES = size * sizeof(int);

    // Define the number of streams and divide the workload
    const int NUM_STREAMS = 8;
    int ELEMENTS_PER_STREAM = size / NUM_STREAMS; // Elements handled by each stream
    int BYTES_PER_STREAM = NO_BYTES / NUM_STREAMS; // Bytes handled by each stream

    // Host pointers
    int *h_a, *h_b, *gpu_result, *cpu_result;

    // Allocate pinned memory for host arrays (faster transfers)
    cudaMallocHost((void**)&h_a, NO_BYTES);
    cudaMallocHost((void**)&h_b, NO_BYTES);
    cudaMallocHost((void**)&gpu_result, NO_BYTES);

    // Allocate memory for CPU result array
    cpu_result = (int *)malloc(NO_BYTES);

    // Initialize host arrays with random values
    initialize(h_a, size);
    initialize(h_b, size);

    // Perform the summation on the CPU
    sumArraysOnHostx(h_a, h_b, cpu_result, size);

    // Device pointers
    int *d_a, *d_b, *d_c;

    // Allocate device memory
    cudaMalloc((int **)&d_a, NO_BYTES);
    cudaMalloc((int **)&d_b, NO_BYTES);
    cudaMalloc((int **)&d_c, NO_BYTES);

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Kernel launch parameters
    dim3 block(block_size); // Threads per block
    dim3 grid(ELEMENTS_PER_STREAM / block.x + 1); // Blocks per grid

    // Offset for partitioning the arrays
    int offset = 0;

    // Launch kernels and memory transfers in streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        offset = i * ELEMENTS_PER_STREAM;

        // Asynchronous memory transfers from host to device
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], BYTES_PER_STREAM, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_b[offset], &h_b[offset], BYTES_PER_STREAM, cudaMemcpyHostToDevice, streams[i]);

        // Asynchronous kernel execution
        sum_array_overlap<<<grid, block, 0, streams[i]>>>(&d_a[offset], &d_b[offset], &d_c[offset], ELEMENTS_PER_STREAM);

        // Asynchronous memory transfers from device to host
        cudaMemcpyAsync(&gpu_result[offset], &d_c[offset], BYTES_PER_STREAM, cudaMemcpyDeviceToHost, streams[i]);
    }

    // Destroy streams after usage
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    // Synchronize device to ensure all operations are complete
    cudaDeviceSynchronize();

    // Validate GPU results against CPU results
    bool match = true;
    for (int i = 0; i < size; i++) {
        if (cpu_result[i] != gpu_result[i]) {
            match = false;
            printf("Mismatch at index %d: CPU %d, GPU %d\n", i, cpu_result[i], gpu_result[i]);
            break;
        }
    }
    if (match) {
        printf("All results match!\n");
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(gpu_result);
    free(cpu_result);

    // Reset the device to clean up resources
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
