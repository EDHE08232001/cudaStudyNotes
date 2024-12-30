# **Matrix Transpose with CUDA and Unrolling**

This project demonstrates how to perform a matrix transpose on the GPU using CUDA. The program includes a kernel that leverages **loop unrolling** for improved memory access efficiency. The implementation highlights performance considerations for global memory access patterns in CUDA.

---

## **Overview**

The program transposes a 1024x1024 matrix using CUDA. It implements the following:
- **Matrix Initialization**: The matrix is filled with integers cycling from 1 to 10 for easy verification.
- **Kernel Execution**: A CUDA kernel performs the transpose using a memory-efficient unrolling technique.
- **Verification**: A portion of the original and transposed matrices is printed for correctness.
- **Memory Management**: The program allocates and frees both host and device memory.

---

## **Key Features**

1. **Loop Unrolling**:
   - Unrolling improves memory throughput by reducing loop overhead and enabling better memory coalescing.
   - Each thread transposes four elements in a single operation.

2. **Optimized Memory Access**:
   - The kernel ensures **coalesced reads and writes**, minimizing memory transaction inefficiencies.

3. **CUDA Best Practices**:
   - Proper use of grid and block dimensions.
   - Synchronization after kernel execution.
   - Efficient memory management on host and device.

---

## **Code Explanation**

### **Kernel: Transpose with Unrolling**
The kernel `transpose_unrolling_read_column_write_row` performs the matrix transpose. Each thread:
1. **Calculates Global Indices**:
   - `ix`: The global x-index of the thread, adjusted for unrolling.
   - `iy`: The global y-index of the thread.

2. **Processes Four Elements**:
   - Each thread handles four contiguous elements from the input matrix using:
     ```cpp
     transpose[to] = mat[ti];
     transpose[to + ny * blockDim.x] = mat[ti + blockDim.x];
     transpose[to + ny * 2 * blockDim.x] = mat[ti + 2 * blockDim.x];
     transpose[to + ny * 3 * blockDim.x] = mat[ti + 3 * blockDim.x];
     ```

3. **Bounds Checking**:
   - Ensures that the indices are within matrix dimensions.

### **Host Code**
1. **Matrix Initialization**:
   - The `initialize` function fills the matrix with numbers 1–10 in a cyclic manner.

2. **Grid and Block Configuration**:
   - Blocks have 128 threads in x and 8 threads in y.
   - Grid dimensions account for unrolling, dividing `nx` by `4 * blockDim.x`.

3. **Memory Management**:
   - Host memory is allocated using `malloc`.
   - Device memory is allocated using `cudaMalloc`.

4. **Verification**:
   - The first 10 elements of the original and transposed matrices are printed to verify correctness.

---

## **How to Run**

### **Requirements**
- CUDA-capable GPU.
- CUDA Toolkit installed.
- Compiler supporting CUDA (e.g., `nvcc`).

### **Compilation**
Compile the program using `nvcc`:
```bash
nvcc -o matrix_transpose matrix_transpose.cu
```

### **Execution**
Run the program:
```bash
./matrix_transpose
```

---

## **Expected Output**

The program will output the first 10 elements of the original matrix and the transposed matrix:
```plaintext
Matrix transpose with unrolling for 1024 by 1024 matrix with block size 128 by 8 

Original Matrix (first 10 elements):
1 2 3 4 5 6 7 8 9 10 

Transposed Matrix with Unrolling (first 10 elements):
1 11 21 31 41 51 61 71 81 91 
```

---

## **Key Takeaways**

1. **Memory Coalescing**:
   - The unrolling technique improves memory coalescing, reducing memory transaction inefficiencies.

2. **Performance**:
   - By processing four elements per thread, the kernel achieves higher memory throughput compared to naive implementations.

3. **CUDA Programming Practices**:
   - Proper use of thread and block indexing, unrolling, and bounds checking is essential for efficient GPU programming.

---

## **Further Improvements**
- Experiment with different matrix sizes and block configurations.
- Extend the implementation to support non-square matrices.
- Profile the kernel execution time using tools like Nsight Compute to quantify performance gains.
