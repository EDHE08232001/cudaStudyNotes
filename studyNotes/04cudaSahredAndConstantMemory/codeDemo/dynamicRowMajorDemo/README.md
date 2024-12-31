# CUDA Dynamic Shared Memory: Row-Major vs Column-Major Access

This repository demonstrates the use of **dynamic shared memory** in CUDA to explore the effects of row-major and column-major access patterns. The program includes a CUDA kernel that writes to shared memory in row-major format and reads it back in column-major format, showcasing how memory layout affects performance and output.

---

## **Key Concepts**

### **Dynamic Shared Memory**
- Shared memory in CUDA is an on-chip memory shared among threads in the same block.
- **Dynamic shared memory** is allocated at runtime using the `extern __shared__` keyword. Its size is specified as a parameter during kernel launch.

### **Row-Major vs Column-Major Access**
- **Row-Major Access**: Memory is accessed sequentially by rows, where elements in a row are contiguous in memory.
- **Column-Major Access**: Memory is accessed sequentially by columns, where elements in a column are contiguous in memory.
- Access patterns impact memory transactions due to the shared memory bank structure.

---

## **Code Overview**

The code demonstrates a kernel function, `setRowReadColDynamic`, which:
1. Allocates dynamic shared memory at runtime.
2. Writes to shared memory in row-major format.
3. Reads from shared memory in column-major format.
4. Writes the result to global memory.

---

## **File Structure**

- **`dynamic_shared_memory_demo.cu`**: Contains the CUDA kernel and host code to demonstrate dynamic shared memory usage.

---

## **Code Highlights**

### **Kernel: setRowReadColDynamic**
```cpp
__global__ void setRowReadColDynamic(int* out) {
    extern __shared__ int tile[];

    int row_index = threadIdx.y * blockDim.x + threadIdx.x;
    int col_index = threadIdx.x * blockDim.x + threadIdx.y;

    // Write to shared memory in row-major format
    tile[row_index] = row_index;

    // Synchronize all threads
    __syncthreads();

    // Read from shared memory in column-major format
    out[row_index] = tile[col_index];
}
```

- **Dynamic Memory Declaration**:
  - `extern __shared__ int tile[];` declares a shared memory array whose size is specified at kernel launch.
- **Index Calculations**:
  - `row_index`: Maps thread indices for row-major writes.
  - `col_index`: Maps thread indices for column-major reads.
- **Synchronization**:
  - `__syncthreads()` ensures all threads complete their writes before any reads.

### **Host Code**
- Allocates memory on the host and device.
- Launches the kernel with dynamically allocated shared memory.
- Copies the results back to the host for verification.

---

## **Execution Instructions**

### **Requirements**
- NVIDIA GPU with CUDA support.
- CUDA Toolkit installed.

### **Compilation**
Compile the program using `nvcc`:
```bash
nvcc dynamic_shared_memory_demo.cu -o dynamic_shared_memory_demo
```

### **Run the Program**
Execute the compiled binary:
```bash
./dynamic_shared_memory_demo
```

---

## **Expected Output**

For a `32x32` matrix, the output demonstrates the effect of row-major writes and column-major reads:

```
=== Output Matrix ===
   0   32   64   96  ...
   1   33   65   97  ...
   2   34   66   98  ...
   ...
```

### **Explanation**
- Each row in the output matrix represents the values stored in shared memory after reading in column-major order and writing in row-major order.
- This illustrates the transformation caused by changing access patterns.

---

## **Key Takeaways**

1. **Dynamic Shared Memory**:
   - Provides flexibility in memory allocation, with size determined at kernel launch.
   - Useful for scenarios with varying memory requirements.

2. **Access Patterns**:
   - Row-major access aligns with memory banks, avoiding conflicts.
   - Column-major access may lead to bank conflicts if not handled carefully.

3. **Performance Optimization**:
   - Synchronization (`__syncthreads()`) is crucial to avoid race conditions.
   - Proper understanding of memory layout improves kernel efficiency.