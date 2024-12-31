#include <stdio.h>
#include <cuda_runtime.h>

/**
 * Kernel: warpReduceSum
 * - Performs a warp-level reduction using __shfl_down_sync.
 * - Reduces an array of integers by summing up all elements.
 *
 * Parameters:
 * - input: Pointer to the input array in global memory.
 * - output: Pointer to the output location in global memory (for final sum).
 * - size: Number of elements in the input array.
 */
__global__ void warpReduceSum(int* input, int* output, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // Global thread ID
    int lane = threadIdx.x % 32;                    // Lane ID within warp (0-31)
    int warp_id = threadIdx.x / 32;                 // Warp ID within block

    // Initialize sum with the input value if within bounds, otherwise set to 0
    int sum = (tid < size) ? input[tid] : 0;

    // Perform warp reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset /= 2) {
        // Each thread adds the value from the thread 'offset' positions away
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset, 32);
    }

    // Write the result of the warp reduction to global memory
    // Only the first thread (lane 0) of each warp writes the result
    if (lane == 0) {
        atomicAdd(output, sum); // Atomic operation to safely accumulate results
    }
}

/**
 * Main function
 * - Initializes the input array.
 * - Allocates memory on the host and device.
 * - Copies data to the device.
 * - Launches the warpReduceSum kernel.
 * - Retrieves and verifies the result.
 */
int main() {
    const int size = 1024;              // Number of elements in the input array
    const int bytes = size * sizeof(int); // Size of the array in bytes

    // Allocate host memory
    int* h_input = (int*)malloc(bytes); // Input array on host
    int h_output = 0;                   // Output sum on host

    // Initialize the input array with values (e.g., all 1s for simplicity)
    for (int i = 0; i < size; ++i) {
        h_input[i] = 1;
    }

    // Allocate device memory
    int* d_input;
    int* d_output;
    cudaMalloc(&d_input, bytes);          // Device input array
    cudaMalloc(&d_output, sizeof(int));  // Device output (single sum value)

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Initialize output memory on the device to 0
    cudaMemcpy(d_output, &h_output, sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    const int threads = 256; // Threads per block
    const int blocks = (size + threads - 1) / threads; // Number of blocks required

    // Launch the warpReduceSum kernel
    warpReduceSum<<<blocks, threads>>>(d_input, d_output, size);
    cudaDeviceSynchronize(); // Ensure kernel execution completes

    // Copy the result back to the host
    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // Verify the result
    printf("Input array initialized with 1s.\n");
    printf("Expected Sum: %d\n", size);
    printf("Computed Sum: %d\n", h_output);

    // Free allocated memory
    free(h_input);            // Free host memory
    cudaFree(d_input);        // Free device input memory
    cudaFree(d_output);       // Free device output memory

    return 0;
}