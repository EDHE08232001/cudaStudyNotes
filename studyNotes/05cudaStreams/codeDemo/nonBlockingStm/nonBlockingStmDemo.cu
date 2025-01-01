#include <stdio.h>
#include <cuda_runtime.h>

// A simple kernel that assigns a value to each element in an array
__global__ void simpleKernel(int *arr, int value, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        arr[idx] = value;
    }
}

int main() {
    const int N = 1024;         // Number of elements
    const int SIZE = N * sizeof(int); // Size of memory in bytes

    // Host and device arrays
    int *h_array, *d_array;
    cudaMallocHost(&h_array, SIZE);     // Allocate pinned host memory
    cudaMalloc(&d_array, SIZE);         // Allocate device memory

    // Initialize host array
    for (int i = 0; i < N; i++) {
        h_array[i] = 0;
    }

    // Create streams
    cudaStream_t blockingStream, nonBlockingStream;
    cudaStreamCreate(&blockingStream); // Default blocking stream
    cudaStreamCreateWithFlags(&nonBlockingStream, cudaStreamNonBlocking); // Non-blocking stream

    // Transfer data to the device using the NULL stream (synchronized with all blocking streams)
    printf("Starting transfer in NULL stream...\n");
    cudaMemcpyAsync(d_array, h_array, SIZE, cudaMemcpyHostToDevice, 0); // NULL stream

    // Launch kernel in a blocking stream
    printf("Launching kernel in a blocking stream...\n");
    simpleKernel<<<N / 256, 256, 0, blockingStream>>>(d_array, 1, N);

    // Launch kernel in a non-blocking stream
    printf("Launching kernel in a non-blocking stream...\n");
    simpleKernel<<<N / 256, 256, 0, nonBlockingStream>>>(d_array, 2, N);

    // Wait for the blocking stream
    printf("Waiting for the blocking stream...\n");
    cudaStreamSynchronize(blockingStream);

    // Wait for the non-blocking stream
    printf("Waiting for the non-blocking stream...\n");
    cudaStreamSynchronize(nonBlockingStream);

    // Copy the results back to the host
    cudaMemcpy(h_array, d_array, SIZE, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < 10; i++) { // Only print the first 10 elements
        printf("h_array[%d] = %d\n", i, h_array[i]);
    }

    // Cleanup
    cudaFree(d_array);
    cudaFreeHost(h_array);
    cudaStreamDestroy(blockingStream);
    cudaStreamDestroy(nonBlockingStream);

    return 0;
}