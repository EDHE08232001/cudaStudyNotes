# **CUDA Streams Overlap Example**

This project demonstrates the use of **CUDA streams** to perform overlapping memory transfers and kernel executions for improved GPU utilization. It showcases how to divide workloads into multiple streams and execute them concurrently, reducing overall execution time and demonstrating the power of CUDA's asynchronous operations.

---

## **What Does This Code Do?**

The program performs an **element-wise addition** of two large arrays (`A` and `B`) to produce a third array (`C`):
- **CPU Computation:** The sum is calculated sequentially on the host for validation.
- **GPU Computation:** The same summation is executed on the GPU using multiple CUDA streams to achieve concurrency.

### **Key Features:**
1. Splits the computation into multiple **streams** for parallel execution.
2. Demonstrates **overlapping** of memory transfers and kernel execution.
3. Validates GPU results by comparing them with the CPU results.

---

## **What Concept Does It Demonstrate?**

This code demonstrates the concept of **CUDA streams** and their ability to enable **asynchronous operations** on the GPU. Specifically:

1. **Asynchronous Memory Transfers:**
   - Transfers data between host (CPU) and device (GPU) memory without blocking the host thread.

2. **Concurrent Execution:**
   - Uses multiple streams to overlap memory transfers and kernel execution, allowing the GPU to process multiple tasks simultaneously.

3. **Improved Resource Utilization:**
   - By keeping the GPU busy with concurrent tasks, idle time is minimized, leading to better performance.

---

## **How to Use This Concept?**

### **When to Use CUDA Streams:**
- When performing **repeated or batched operations** on large datasets.
- To achieve **real-time processing** by overlapping data transfer and computation.
- For **multi-kernel workloads** where different kernels can execute concurrently.

### **Steps to Use CUDA Streams:**
1. **Create Streams:**
   ```cpp
   cudaStream_t stream;
   cudaStreamCreate(&stream);
   ```
2. **Assign Tasks to Streams:**
   - Use `cudaMemcpyAsync` for asynchronous memory transfers.
   - Launch kernels in specific streams by passing the stream as the fourth kernel argument.
   ```cpp
   cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream);
   kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, size);
   ```
3. **Synchronize Streams:**
   - Use `cudaStreamSynchronize` to wait for all operations in a stream to complete.
   ```cpp
   cudaStreamSynchronize(stream);
   ```
4. **Destroy Streams:**
   ```cpp
   cudaStreamDestroy(stream);
   ```

---

## **How Does This Code Work?**

### **Overview of the Workflow:**
1. **Initialization:**
   - Host arrays (`A`, `B`, `C`) are initialized with random values.
   - Device memory is allocated for arrays `A`, `B`, and `C`.

2. **Stream Creation:**
   - Eight CUDA streams are created to divide the workload.

3. **Workload Division:**
   - The large arrays are partitioned into smaller chunks, each processed by a separate stream.

4. **Asynchronous Operations:**
   - Each stream performs the following steps **concurrently**:
     1. Transfer a chunk of `A` and `B` from host to device memory.
     2. Perform element-wise addition on the GPU.
     3. Transfer the result back to host memory.

5. **Synchronization:**
   - The host waits for all streams to finish before proceeding.

6. **Validation:**
   - The GPU result is compared with the CPU result for correctness.

7. **Cleanup:**
   - All allocated memory and streams are freed to prevent resource leaks.

---

## **How to Run the Code**

### **Prerequisites:**
1. A system with a CUDA-capable GPU.
2. CUDA Toolkit installed and properly configured.

### **Steps:**
1. Clone the repository or save the files.
2. Compile the program using the NVIDIA CUDA Compiler (`nvcc`):
3. Run the executable:

### **Expected Output:**
- The program will print "All results match!" if the GPU computation is correct.
- If there is a mismatch, the program will report the first index where the results differ.

---

## **Educational Insights**

### **What You Will Learn:**
1. **Concepts of Asynchronous Programming:**
   - How to use CUDA streams to overlap data transfers and kernel execution.
2. **Performance Optimization:**
   - Reducing overall execution time by keeping the GPU busy with concurrent operations.
3. **Resource Management:**
   - Best practices for allocating, synchronizing, and cleaning up CUDA streams and memory.

### **Experimentation Ideas:**
- **Change the Number of Streams:**
  - Experiment with different values of `NUM_STREAMS` to observe the impact on performance.
- **Measure Execution Time:**
  - Use CUDA events to measure the time taken for GPU computations.
- **Increase Problem Size:**
  - Increase `size` to observe how CUDA streams scale with larger workloads.

-----

# Understanding the Offset in CUDA Streams

The **offset** in this program is used to partition the workload across multiple streams. Each stream processes a distinct portion (or chunk) of the arrays, ensuring that the operations do not overlap.

---

### **What is the Offset?**
The **offset** determines the starting index in the array for each stream's portion of the workload. It ensures that:
1. Each stream operates on a separate chunk of the array.
2. Memory transfers and kernel executions are properly partitioned for concurrent processing.

---

### **How Does Offset Work in This Code?**

1. **Total Elements in Array:**
   - The array contains `size` elements.
2. **Number of Streams:**
   - The workload is divided into `NUM_STREAMS` equal parts.
3. **Elements Per Stream:**
   - Each stream processes `ELEMENTS_PER_STREAM = size / NUM_STREAMS`.
4. **Offset for Each Stream:**
   - For stream `i`, the starting index is:
     ```cpp
     offset = i * ELEMENTS_PER_STREAM;
     ```
   - This ensures that stream `i` works on indices `[offset, offset + ELEMENTS_PER_STREAM)`.

---

### **Example: Visualizing Offset**

Suppose:
- `size = 16` (16 elements in the array).
- `NUM_STREAMS = 4` (4 streams).
- `ELEMENTS_PER_STREAM = size / NUM_STREAMS = 4`.

#### **Workload Partitioning with Offset**
| **Stream** | **Offset** | **Indices Processed** | **Array Elements**        |
|------------|------------|-----------------------|---------------------------|
| Stream 0   | `0`        | `[0, 1, 2, 3]`       | `A[0] A[1] A[2] A[3]`     |
| Stream 1   | `4`        | `[4, 5, 6, 7]`       | `A[4] A[5] A[6] A[7]`     |
| Stream 2   | `8`        | `[8, 9, 10, 11]`     | `A[8] A[9] A[10] A[11]`   |
| Stream 3   | `12`       | `[12, 13, 14, 15]`   | `A[12] A[13] A[14] A[15]` |

Each stream processes its own chunk independently.

---

### **Diagram Representation**
Below is a textual representation of how the array is divided and processed by the streams:

```
Array A (Host Memory): [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15]

Stream 0:
  - Offset: 0
  - Indices: [0, 1, 2, 3]
  - Elements: [A0, A1, A2, A3]
  - Operations: Transfer Host → Device, Kernel Execution, Transfer Device → Host

Stream 1:
  - Offset: 4
  - Indices: [4, 5, 6, 7]
  - Elements: [A4, A5, A6, A7]
  - Operations: Transfer Host → Device, Kernel Execution, Transfer Device → Host

Stream 2:
  - Offset: 8
  - Indices: [8, 9, 10, 11]
  - Elements: [A8, A9, A10, A11]
  - Operations: Transfer Host → Device, Kernel Execution, Transfer Device → Host

Stream 3:
  - Offset: 12
  - Indices: [12, 13, 14, 15]
  - Elements: [A12, A13, A14, A15]
  - Operations: Transfer Host → Device, Kernel Execution, Transfer Device → Host
```

---

### **Key Insights**
1. **Independent Chunks:**  
   Each stream operates on a non-overlapping section of the array. This independence ensures that memory transfers and computations can happen concurrently.

2. **Concurrent Execution:**  
   Streams execute their respective tasks (memory transfers, kernel execution) at the same time, provided the GPU has sufficient resources.

3. **Offset Calculation:**  
   The offset is a simple way to divide the workload:
   ```cpp
   offset = i * ELEMENTS_PER_STREAM;
   ```

---

### **How Offset Works in Code**
```cpp
for (int i = 0; i < NUM_STREAMS; i++) {
    offset = i * ELEMENTS_PER_STREAM; // Calculate the starting index for this stream

    // Asynchronous memory transfers
    cudaMemcpyAsync(&d_a[offset], &h_a[offset], BYTES_PER_STREAM, cudaMemcpyHostToDevice, streams[i]);
    cudaMemcpyAsync(&d_b[offset], &h_b[offset], BYTES_PER_STREAM, cudaMemcpyHostToDevice, streams[i]);

    // Kernel execution
    sum_array_overlap<<<grid, block, 0, streams[i]>>>(&d_a[offset], &d_b[offset], &d_c[offset], ELEMENTS_PER_STREAM);

    // Asynchronous memory transfer back to host
    cudaMemcpyAsync(&gpu_result[offset], &d_c[offset], BYTES_PER_STREAM, cudaMemcpyDeviceToHost, streams[i]);
}
```

---

### **Benefits of Using Offset**
- **Partitioning:** Simplifies dividing the workload among multiple streams.
- **Scalability:** Easy to adjust for larger arrays or more streams.
- **Concurrent Execution:** Enables multiple streams to work on separate chunks concurrently.