#include <stdio.h>
#include <cuda_runtime.h>

// Declare constant memory
__constant__ float coefficients[5];

// Kernel to perform stencil computation using constant memory
__global__ void stencilKernel(const float* input, float* output, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure threads access only valid data
    if (idx >= 2 && idx < size - 2) {
        output[idx] = coefficients[0] * input[idx - 2] +
                      coefficients[1] * input[idx - 1] +
                      coefficients[2] * input[idx] +
                      coefficients[3] * input[idx + 1] +
                      coefficients[4] * input[idx + 2];
    }
}

int main() {
    const int size = 1024;            // Size of the input and output arrays
    const int bytes = size * sizeof(float); // Memory size in bytes

    // Host arrays
    float* h_input = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    float h_coefficients[5] = {0.1f, 0.2f, 0.4f, 0.2f, 0.1f}; // Stencil coefficients

    // Initialize input array with some values
    for (int i = 0; i < size; ++i) {
        h_input[i] = (float)i;
    }

    // Device arrays
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Copy coefficients to constant memory
    cudaMemcpyToSymbol(coefficients, h_coefficients, sizeof(h_coefficients));

    // Define grid and block dimensions
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);

    // Launch the kernel
    stencilKernel<<<grid, block>>>(d_input, d_output, size);
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Verify the result (print first 10 values)
    printf("Input: ");
    for (int i = 0; i < 10; ++i) {
        printf("%.1f ", h_input[i]);
    }
    printf("\nOutput: ");
    for (int i = 0; i < 10; ++i) {
        printf("%.1f ", h_output[i]);
    }
    printf("\n");

    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
