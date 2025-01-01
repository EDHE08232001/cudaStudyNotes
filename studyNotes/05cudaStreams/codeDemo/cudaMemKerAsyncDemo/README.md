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
   ```bash
   nvcc -o cuda_stream_demo cuda_stream_demo.cu
   ```
3. Run the executable:
   ```bash
   ./cuda_stream_demo
   ```

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