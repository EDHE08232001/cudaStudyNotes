#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/**
 * CUDA kernel function for demonstrating asynchronous streams.
 * This kernel performs a simple computation on each element of the input array.
 * 
 * @param in - Pointer to input array on the device.
 * @param out - Pointer to output array on the device.
 * @param size - Number of elements in the input and output arrays.
 */
__global__ void cuda_stream_async_demo(int* in, int* out, int size) {
    // Calculate the global thread ID
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    // Perform computation only if the thread ID is within bounds
    if (gid < size) {
        // Simulate a computational workload for demonstration
        for (int i = 0; i < 25; i++) {
            out[gid] = in[gid] + (in[gid] - 1) * (gid % 10);
        }
    }
}

/**
 * Initializes an array with test data.
 * Each element is initialized to `(i % 10) + 1`.
 *
 * @param arr - Pointer to the array to initialize.
 * @param size - Number of elements in the array.
 */
void initialize(int* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = (i % 10) + 1;
    }
}

int main(int argc, char** argv) {
    // Define the size of the arrays (2^18 elements)
    int size = 1 << 18;
    int byte_size = sizeof(int) * size;

    // Declare host pointers (pinned memory for better performance)
    int* h_in;   // Input array for first stream
    int* h_ref;  // Output array for first stream
    int* h_in2;  // Input array for second stream
    int* h_ref2; // Output array for second stream

    // Allocate pinned host memory (faster data transfers with pinned memory)
    cudaMallocHost((void**)&h_in, byte_size);
    cudaMallocHost((void**)&h_ref, byte_size);
    cudaMallocHost((void**)&h_in2, byte_size);
    cudaMallocHost((void**)&h_ref2, byte_size);

    // Initialize the host arrays
    initialize(h_in, size);
    initialize(h_in2, size);

    // Allocate device memory
    int* d_in;    // Device input for first stream
    int* d_out;   // Device output for first stream
    int* d_in2;   // Device input for second stream
    int* d_out2;  // Device output for second stream
    cudaMalloc((void**)&d_in, byte_size);
    cudaMalloc((void**)&d_out, byte_size);
    cudaMalloc((void**)&d_in2, byte_size);
    cudaMalloc((void**)&d_out2, byte_size);

    // Create CUDA streams for asynchronous operations
    cudaStream_t cuda_stream;
    cudaStream_t cuda_stream2;
    cudaStreamCreate(&cuda_stream);
    cudaStreamCreate(&cuda_stream2);

    // Kernel configuration (128 threads per block, sufficient blocks to cover the array size)
    dim3 block(128);
    dim3 grid(size / block.x);

    // First stream: Async data transfer, kernel execution, and result retrieval
    cudaMemcpyAsync(d_in, h_in, byte_size, cudaMemcpyHostToDevice, cuda_stream);
    cuda_stream_async_demo<<<grid, block, 0, cuda_stream>>>(d_in, d_out, size);
    cudaMemcpyAsync(h_ref, d_out, byte_size, cudaMemcpyDeviceToHost, cuda_stream);

    // Second stream: Async data transfer, kernel execution, and result retrieval
    cudaMemcpyAsync(d_in2, h_in2, byte_size, cudaMemcpyHostToDevice, cuda_stream2);
    cuda_stream_async_demo<<<grid, block, 0, cuda_stream2>>>(d_in2, d_out2, size);
    cudaMemcpyAsync(h_ref2, d_out2, byte_size, cudaMemcpyDeviceToHost, cuda_stream2);

    // Synchronize the device to ensure all operations complete
    cudaDeviceSynchronize();

    // Synchronize and destroy streams
    cudaStreamSynchronize(cuda_stream);
    cudaStreamDestroy(cuda_stream);
    cudaStreamSynchronize(cuda_stream2);
    cudaStreamDestroy(cuda_stream2);

    // Reset the device to clean up all allocations
    cudaDeviceReset();

    return 0;
}
