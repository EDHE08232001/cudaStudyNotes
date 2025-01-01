# **CUDA Streams and Events**

In CUDA programming, we often follow a basic model for structuring program execution:

### **Kernel-Level Parallelism**
```
Data Transfer: Host → Device  -->  Kernel Execution  -->  Data Transfer: Device → Host
--------------------------------------------------------------------------------------> (Time)
```

While effective, this model can be limiting in terms of performance optimization, as the different stages of execution are sequentially dependent.

---

## **Grid-Level Concurrency/Parallelism**

To maximize concurrency, we can execute multiple kernels simultaneously on the same device while overlapping memory transfers with kernel execution.

![Grid Level Concurrency](./images/gridConcurrencyDiagram.webp)

### **Optimized Approach**
What if we:
1. Partition the data.
2. Transfer one partition to the device and execute the kernel on it.
3. Simultaneously transfer the next partition while the kernel is still executing.

This overlapping of memory transfer and kernel execution reduces overall execution time.

### **Requirements for Overlap**
To enable such concurrency:
1. We need a way to **launch multiple kernels** on the same device.
2. We need **asynchronous memory transfer mechanisms**.

---

## **CUDA Streams**

A **stream** is a sequence of commands (kernel executions, memory transfers, etc.) that execute in order.  
Commands in **different streams** can:
- Execute **out of order** relative to each other.
- Execute **concurrently**, depending on hardware capabilities.

---

## **Synchronous vs. Asynchronous Function Calls**

From the **host's perspective**:
- **Synchronous Functions**: Block the host thread until the operation completes.
- **Asynchronous Functions**: Return control to the host immediately after the function call, allowing further operations to proceed.

### Example
```cpp
kernel1<<<grid, block, 0, stream1>>>();
kernel2<<<grid, block>>>();  // Default (null) stream
kernel3<<<grid, block, 0, stream2>>>();
```

#### Observations:
1. Three kernels (`kernel1`, `kernel2`, and `kernel3`) are launched in separate streams (`stream1`, default stream, and `stream2`).
2. From the **host's perspective**, all these launches are **asynchronous**—the host does not wait for them to complete.
3. From the **device's perspective**, execution depends on the relationship between `stream1`, `stream2`, and the null stream:
   - Commands in the null stream **synchronize** with other streams by default.
   - Commands in `stream1` and `stream2` can potentially execute concurrently if the device supports it.

---

## **NULL Stream (Default Stream)**

The **null stream** is the default stream used for kernel launches and data transfers if no stream is explicitly specified. It has unique behavior:
- **Synchronizes** with commands in other streams.
- Often serves as a synchronization mechanism between multiple streams.

---

## **Concurrency-Enabled Tasks in CUDA**

The following tasks can operate concurrently, subject to hardware capabilities:
1. Computations on the **host**.
2. Computations on the **device**.
3. Memory transfers **Host → Device**.
4. Memory transfers **Device → Host**.
5. Memory transfers **within device memory**.
6. Memory transfers **across devices**.

---

### **Concurrency Visualization**
Consider the following overlap model:

```text
Memory Transfer: Host → Device  (Stream 1)        ---------------------
Kernel Execution 1 (Stream 1)                     ---------------------
Memory Transfer: Host → Device  (Stream 2)        ---------------------
Kernel Execution 2 (Stream 2)                     ---------------------
```

Commands in **different streams** can execute concurrently, reducing bottlenecks and optimizing overall performance.

-----

# Asynchronous Functions

To transfer data asynchronously, use
```c
cudaMemCpyAsync(
    destination_pointer,            <- This should be Pinned Memory
    source_pointer,
    size,
    memory_copy_direction,
    stream
);
```

**Note:** To perform asynchronous memeory transfers, CUDA run time need the gurantee that the operating system will not move the virtual memory that belongs to the memory being copied in the middle of the memory transfer operation. Therefore, we have to use pinned memory with the above function. If we used unpinned memory, then this memory transfer will be a synchronous one which block the host execution.

-----

# **How to Use CUDA Streams**

### **Goal**
The primary goal of using CUDA streams is to:
1. **Overlap kernel executions** with memory transfers.
2. **Reduce overall execution time** by improving GPU utilization.

By leveraging CUDA streams, multiple operations (memory transfers, kernel executions) can occur **concurrently**, provided the hardware supports it.

For an implementation example, refer to **`cudaStreamDemo.cu`**.

---

## **Key Steps to Use Streams**

### 1. **Stream Declaration**
   ```cpp
   cudaStream_t stream;
   ```
   - A stream object must be declared using the `cudaStream_t` type.

---

### 2. **Stream Creation**
   ```cpp
   cudaStreamCreate(&stream);
   ```
   - Initializes a stream for use.
   - Every command assigned to this stream will execute in the order they are queued, independently of commands in other streams.

---

### 3. **Stream Synchronization**
   ```cpp
   cudaStreamSynchronize(stream);
   ```
   - Blocks the host program until **all operations** in the specified stream are complete.
   - Useful for ensuring that dependent operations do not proceed until previous stream commands finish.

---

### 4. **Stream Query**
   ```cpp
   cudaError_t status = cudaStreamQuery(stream);
   ```
   - **Non-blocking function** that checks the status of operations in the specified stream.
   - **Return values:**
     - `cudaSuccess`: All operations in the stream are complete.
     - `cudaErrorNotReady`: Operations in the stream are still in progress.

#### **Why Use `cudaStreamQuery`?**
- Allows you to periodically check the stream's status **without blocking host operations**, enabling more efficient utilization of host resources.

---

### 5. **Stream Destruction**
   ```cpp
   cudaStreamDestroy(stream);
   ```
   - Frees resources associated with the stream after all operations in the stream are complete.
   - Essential for preventing resource leaks.

---

## **Benefits of Using Streams**

1. **Asynchronous Memory Transfers**
   - Use `cudaMemcpyAsync` to transfer data between host and device memory in a specific stream.
   - Overlap memory transfers with kernel executions in the same or different streams.

2. **Concurrent Kernel Execution**
   - Launch multiple kernels in different streams to execute concurrently, provided the device supports concurrent kernel execution.

3. **Improved Resource Utilization**
   - Keep the GPU busy by overlapping memory operations with computation, reducing idle time.

---

## **Example Workflow with Streams**
Here’s a simple breakdown of how to use streams effectively:

**Preparation:** Check concurrent kernel execution eligibility
```c
int dev = 0;
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, dev);

if (devicePro.concurrentKernels == 0) {
    printf("Eligible");
}
```

1. **Initialize Streams**:
   ```cpp
   cudaStream_t stream1, stream2;
   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);
   ```

2. **Perform Asynchronous Operations**:
   - Transfer data to device memory and launch kernels:
     ```cpp
     cudaMemcpyAsync(d_in, h_in, size, cudaMemcpyHostToDevice, stream1);
     kernel<<<grid, block, 0, stream1>>>(d_in, d_out, size);
     cudaMemcpyAsync(h_out, d_out, size, cudaMemcpyDeviceToHost, stream1);
     ```

3. **Query Stream Status**:
   ```cpp
   if (cudaStreamQuery(stream1) == cudaErrorNotReady) {
       printf("Stream 1 is still processing.\n");
   }
   ```

4. **Synchronize Streams** (if required):
   ```cpp
   cudaStreamSynchronize(stream1);
   ```

5. **Destroy Streams**:
   ```cpp
   cudaStreamDestroy(stream1);
   cudaStreamDestroy(stream2);
   ```

---

### **Visualizing Stream Behavior**
- **Without Streams**: Sequential execution of memory transfers and kernel operations.
  ```
  [Memory Transfer Host → Device] → [Kernel Execution] → [Memory Transfer Device → Host]
  ```

- **With Streams**: Concurrent execution of operations in separate streams.
  ```
  Stream 1: [Memory Transfer Host → Device] → [Kernel Execution]
  Stream 2:                          [Memory Transfer Device → Host]
  ```

By overlapping these operations, the GPU can process tasks more efficiently, leading to reduced overall execution time.

---

### **Common Use Cases**
1. **Real-Time Data Processing**:
   - Continuous processing of data chunks in different streams.
2. **Multi-Kernel Workloads**:
   - Execute independent kernels simultaneously to utilize the GPU better.
3. **Heterogeneous Workloads**:
   - Overlap memory-bound and compute-bound tasks using streams.