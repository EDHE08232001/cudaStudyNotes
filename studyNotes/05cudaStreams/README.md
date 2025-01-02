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

-----

# **Overlapping Memory Transfer and Kernel Execution**

In CUDA programming, overlapping memory transfer and kernel execution is a crucial optimization technique to improve GPU utilization. By dividing tasks among streams and leveraging asynchronous operations, you can achieve concurrency between data movement and computation.

For a practical demonstration, refer to the folder `cudaMemKerAsyncDemo` in `codeDemo`.

---

# **Stream Synchronization and Blocking Behaviors of the NULL Stream**

---

## **Blocking Behavior of the NULL Stream**

The **NULL stream** is an implicit stream in CUDA. It has a **blocking behavior** that affects how operations in other streams execute:

- **Non-NULL Streams:**
  - While non-NULL streams are generally **non-blocking** with respect to the host, operations in non-NULL streams can be **blocked by the NULL stream**.

- **NULL Stream Behavior:**
  - The NULL stream synchronizes with all **blocking streams** in the same CUDA context.
  - When an operation is issued to the NULL stream, the CUDA context **waits for all operations previously issued to blocking streams** to complete before executing the NULL stream operation.

---

### **Types of Non-NULL Streams**
CUDA streams can be categorized into two types based on their synchronization behavior:

1. **Blocking Streams**:
   - Operations in a blocking stream can be delayed by the NULL stream.
   - **Streams created with `cudaStreamCreate()` are blocking streams by default.**

2. **Non-Blocking Streams**:
   - Operations in a non-blocking stream do not synchronize with the NULL stream.
   - Non-blocking streams can operate independently of the NULL stream.

---

## **Creating Non-Blocking Streams**

To create non-blocking streams, CUDA provides the following API:

```c
cudaStreamCreateWithFlags(
   cudaStream_t* pStream,
   unsigned int flags
);
```

### **Flags:**
- `cudaStreamDefault`:
  - Default behavior for streams.
  - Creates a **blocking stream**.
- `cudaStreamNonBlocking`:
  - Creates a **non-blocking stream** that operates independently of the NULL stream.

### **Example:**
```cpp
cudaStream_t stream1, stream2;

// Create a blocking stream
cudaStreamCreate(&stream1);  // Equivalent to cudaStreamCreateWithFlags(&stream1, cudaStreamDefault);

// Create a non-blocking stream
cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
```

---

### **Why Non-Blocking Streams Are Useful**
Non-blocking streams allow you to:
1. Achieve **concurrent execution** without being affected by the NULL stream's synchronization.
2. Enable fine-grained control over execution dependencies in complex workloads.

---

### **NULL Stream Synchronization Diagram**

The diagram below illustrates how the NULL stream synchronizes with blocking streams:

```
Time --->

NULL Stream:  [---------Op1---------]   [---Op3---]
Blocking Stream 1:  [---Op2---]              [---Op4---]

Op1 in the NULL stream must complete before Op2 in Blocking Stream 1 starts.
Op3 in the NULL stream waits for Op4 in Blocking Stream 1 to finish.
```

---

### **Key Notes:**
- Operations in non-blocking streams are **not synchronized** with the NULL stream and can execute concurrently.
- Use `cudaStreamNonBlocking` to create independent streams for better concurrency.
- The default behavior of `cudaStreamCreate()` creates streams that are synchronized with the NULL stream.

-----

# **Explicit and Implicit Synchronization in CUDA**

CUDA provides mechanisms for both **explicit** and **implicit synchronization** to control the execution flow between the host and device, as well as between streams. Understanding these concepts is critical for optimizing performance and ensuring correctness in CUDA applications.

---

## **Explicit Synchronization**

Explicit synchronization is when the programmer deliberately specifies synchronization points to control execution. These synchronization APIs ensure that certain tasks are completed before others proceed.

### **Common Explicit Synchronization APIs:**

1. **`cudaDeviceSynchronize()`**:
   - Synchronizes the host with the entire device.
   - Blocks the host thread until **all previously issued tasks on the device** (including all streams) are complete.

   **Use Case:** Ensure all GPU tasks are finished before the host accesses the results.

   ```cpp
   cudaDeviceSynchronize();
   ```

2. **`cudaStreamSynchronize()`**:
   - Synchronizes the host with a **specific stream**.
   - Blocks the host thread until all tasks in the specified stream are complete.

   **Use Case:** Ensure tasks in a specific stream are completed before continuing on the host.

   ```cpp
   cudaStreamSynchronize(myStream);
   ```

3. **`cudaEventSynchronize()`**:
   - Synchronizes the host with a **specific event**.
   - Blocks the host thread until the specified event has been recorded in its stream.

   **Use Case:** Synchronize the host with a particular milestone in the GPU’s execution.

   ```cpp
   cudaEventSynchronize(myEvent);
   ```

4. **`cudaStreamWaitEvent()`**:
   - Synchronizes one stream with an **event recorded in another stream**.
   - Ensures that the second stream waits until the event in the first stream is completed.

   **Use Case:** Create dependencies between streams.

   ```cpp
   cudaStreamWaitEvent(myStream, myEvent, 0);
   ```

---

## **Implicit Synchronization**

Implicit synchronization occurs **automatically as a side effect** of certain CUDA operations. These operations introduce a synchronization point without the programmer explicitly calling a synchronization API.

### **Common Causes of Implicit Synchronization:**

1. **Blocking Function Calls:**
   - Blocking functions halt the host until certain conditions are met.
   - Example: **`cudaMemcpy`** (synchronous version) implicitly synchronizes the device.

2. **Operations on the NULL Stream:**
   - Any task issued to the NULL stream will synchronize with all blocking streams.
   - **Example:** Launching a kernel in the NULL stream waits for all tasks in blocking streams to complete.

3. **Memory Operations:**
   - Certain memory operations implicitly synchronize the device:
     - **Host-Device Memory Transfers:**
       - Example: `cudaMemcpy` between host and device memory.
     - **Device Memory Allocation:**
       - Example: `cudaMalloc` and `cudaFree`.
     - **Memory Initialization:**
       - Example: `cudaMemset`.

4. **Switching Configurations:**
   - Changing configurations like the **L1 cache/shared memory preference** introduces an implicit synchronization point.
   - Example: `cudaDeviceSetCacheConfig`.

5. **Page-Locked Host Memory Allocation:**
   - Allocating pinned memory on the host with `cudaMallocHost` also introduces an implicit synchronization point.

---

### **Explicit vs. Implicit Synchronization**

| **Aspect**                | **Explicit Synchronization**                        | **Implicit Synchronization**                 |
|---------------------------|----------------------------------------------------|---------------------------------------------|
| **Control**               | Programmer explicitly defines synchronization points. | Automatic and not explicitly controlled.    |
| **Granularity**           | Device, stream, or event level.                    | Depends on the operation (e.g., memory or stream). |
| **Performance Impact**    | Requires careful placement to avoid unnecessary delays. | May introduce hidden bottlenecks.           |
| **Use Case**              | Ensuring correctness for specific execution dependencies. | Happens as a side effect of certain operations. |

---

### **Examples of Synchronization Scenarios**

1. **Explicit Synchronization:**
   ```cpp
   // Create a stream and an event
   cudaStream_t stream;
   cudaEvent_t event;
   cudaStreamCreate(&stream);
   cudaEventCreate(&event);

   // Launch a kernel in the stream
   myKernel<<<blocks, threads, 0, stream>>>();

   // Record an event in the stream
   cudaEventRecord(event, stream);

   // Synchronize host with the event
   cudaEventSynchronize(event);

   // Cleanup
   cudaStreamDestroy(stream);
   cudaEventDestroy(event);
   ```

2. **Implicit Synchronization:**
   ```cpp
   // Allocate device memory
   int *d_array;
   cudaMalloc(&d_array, size);  // Implicit synchronization point

   // Copy memory (synchronous version)
   cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);  // Implicit synchronization

   // Launch a kernel in the NULL stream
   myKernel<<<blocks, threads>>>();  // Implicitly synchronized with blocking streams
   ```

---

### **Key Takeaways**
1. **Explicit Synchronization** provides precise control but should be used sparingly to avoid performance bottlenecks.
2. **Implicit Synchronization** occurs automatically and may lead to unintended delays if not understood properly.
3. For optimal performance, combine **non-blocking streams**, **asynchronous operations**, and **minimal synchronization** wherever possible.

-----

# **CUDA Events and Timing with CUDA Events**

CUDA events are powerful tools for **synchronization** and **performance measurement** in CUDA applications. They allow you to mark specific points in the execution flow and monitor the progress of device tasks.

---

## **What is a CUDA Event?**

A **CUDA event** is a marker in a CUDA stream that represents a point in the execution timeline of operations in that stream. Events are used for:
1. **Synchronization**: Coordinate tasks across multiple streams or between the host and device.
2. **Progress Monitoring**: Check the status of tasks in a stream.
3. **Performance Measurement**: Accurately measure the execution time of CUDA operations.

---

## **Key Properties of CUDA Events**

1. **Stream Association**:
   - An event recorded in a stream will be considered "satisfied" only after all preceding operations in that stream are complete.

2. **Default Stream Behavior**:
   - Events recorded on the **NULL (default) stream** are applied globally, meaning they synchronize with all streams in the same CUDA context.

3. **Blocking vs. Non-Blocking**:
   - Events can either block the host thread (`cudaEventSynchronize`) or allow non-blocking checks (`cudaEventQuery`).

---

## **CUDA Event API**

### 1. **Event Creation**
   - Create an event using `cudaEventCreate`:
     ```cpp
     cudaEvent_t event;
     cudaEventCreate(&event);
     ```
   - Events can also be created with flags to modify their behavior:
     - `cudaEventDefault`: Default event behavior.
     - `cudaEventBlockingSync`: Synchronizes the host thread when the event is queried or waited upon.
     - `cudaEventDisableTiming`: Creates an event without timing capabilities, useful for pure synchronization.

### 2. **Event Recording**
   - Record an event in a specific stream using `cudaEventRecord`:
     ```cpp
     cudaEventRecord(event, stream);
     ```
   - The event is added to the execution queue of the specified stream.

### 3. **Event Synchronization**
   - Block the host thread until the event is completed:
     ```cpp
     cudaEventSynchronize(event);
     ```

### 4. **Event Query**
   - Check if an event has completed without blocking the host:
     ```cpp
     cudaError_t status = cudaEventQuery(event);
     if (status == cudaSuccess) {
         // Event has completed
     } else {
         // Event is not yet complete
     }
     ```

### 5. **Event Timing**
   - Measure the elapsed time between two events using `cudaEventElapsedTime`:
     ```cpp
     float milliseconds = 0;
     cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
     ```
   - This measures the time (in milliseconds) between when `startEvent` and `stopEvent` were recorded.

### 6. **Event Destruction**
   - Free the resources associated with an event:
     ```cpp
     cudaEventDestroy(event);
     ```

---

## **Example: Using CUDA Events for Timing**

Here’s a simple example demonstrating how to use CUDA events for timing a kernel execution:

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel
__global__ void simpleKernel(float *arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        arr[idx] = idx * 2.0f;
    }
}

int main() {
    const int N = 1 << 20; // Number of elements
    const int SIZE = N * sizeof(float);

    // Host and device arrays
    float *h_array, *d_array;
    cudaMallocHost(&h_array, SIZE); // Pinned memory for host
    cudaMalloc(&d_array, SIZE);    // Device memory

    // Initialize host array
    for (int i = 0; i < N; i++) {
        h_array[i] = 0.0f;
    }

    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start recording the timing
    cudaEventRecord(start, 0); // Record start event in the default stream

    // Copy data to device
    cudaMemcpy(d_array, h_array, SIZE, cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    simpleKernel<<<grid, block>>>(d_array, N);

    // Copy data back to host
    cudaMemcpy(h_array, d_array, SIZE, cudaMemcpyDeviceToHost);

    // Stop recording the timing
    cudaEventRecord(stop, 0); // Record stop event in the default stream

    // Synchronize to ensure all operations are complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the elapsed time
    printf("Elapsed time: %.2f ms\n", milliseconds);

    // Cleanup
    cudaFree(d_array);
    cudaFreeHost(h_array);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```

---

## **Key Points in Example**

1. **Events for Timing**:
   - `start` and `stop` events are recorded before and after the CUDA operations.
   - `cudaEventElapsedTime` measures the time between the two events.

2. **Synchronization**:
   - `cudaEventSynchronize(stop)` ensures all operations are completed before measuring the time.

3. **Accuracy**:
   - The timing is accurate to the level of individual CUDA operations in the same stream.

---

## **Use Cases of CUDA Events**

1. **Synchronization**:
   - Coordinate tasks across multiple streams using `cudaEventSynchronize` or `cudaStreamWaitEvent`.

2. **Performance Measurement**:
   - Accurately measure execution times of kernels, memory transfers, or any CUDA operations.

3. **Debugging**:
   - Use `cudaEventQuery` to monitor the progress of operations in a stream without blocking the host.

---

## **Best Practices**

1. **Use Events for Timing**:
   - Prefer `cudaEventElapsedTime` for measuring GPU timings instead of host-side timing functions, as it provides more accurate results.

2. **Avoid Excessive Synchronization**:
   - Synchronizing too frequently (e.g., `cudaEventSynchronize` after every operation) can harm performance by reducing concurrency.

3. **Use Non-Timing Events for Pure Synchronization**:
   - Create events with `cudaEventDisableTiming` for synchronization-only purposes to reduce overhead.

-----

# **Creating Inter-Stream Dependencies with Events**

In CUDA, **inter-stream dependencies** allow you to coordinate operations between multiple streams, ensuring that tasks in one stream do not proceed until tasks in another stream have completed. This is especially useful in complex applications where different streams need to work together in a controlled manner.

---

## **Using Events for Inter-Stream Dependencies**

CUDA **events** can be used to introduce inter-stream dependencies. These events act as synchronization points between streams.

### **Creating Events with Flags**

You can create events with specific properties using:

```c
cudaEventCreateWithFlags(
   cudaEvent_t* event,
   unsigned int flags
);
```

### **Common Flags and Their Behavior**
- **`cudaEventDefault`**:
   - Default event behavior.
   - Records timing data and synchronizes normally.
- **`cudaEventBlockingSync`**:
   - Ensures that `cudaEventSynchronize()` blocks the calling thread until the event is completed.
- **`cudaEventDisableTiming`**:
   - Creates an event that does not record timing information.
   - Useful for pure synchronization without the overhead of timing.
- **`cudaEventInterprocess`**:
   - Allows the event to be used across processes (e.g., for multi-process CUDA applications).

### **Example: Creating an Event**
```cpp
cudaEvent_t event;
cudaEventCreateWithFlags(&event, cudaEventDefault); // Create an event with default behavior
```

---

## **Creating Inter-Stream Dependencies**

To create a dependency where one stream waits for an event recorded in another stream, use:

```c
cudaStreamWaitEvent(
   cudaStream_t stream,
   cudaEvent_t event
);
```

### **How It Works**
1. **Record an Event in Stream A**:
   - Use `cudaEventRecord()` to queue an event at a specific point in `Stream A`.
2. **Make Stream B Wait for the Event**:
   - Use `cudaStreamWaitEvent()` to ensure `Stream B` waits for the event in `Stream A` to complete before proceeding.

---

## **Example: Inter-Stream Dependencies**

This example demonstrates how to create inter-stream dependencies between two streams:

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

// A simple kernel
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
    cudaMallocHost(&h_array, SIZE); // Pinned memory for host
    cudaMalloc(&d_array, SIZE);    // Device memory

    // Initialize host array
    for (int i = 0; i < N; i++) {
        h_array[i] = 0;
    }

    // Create two streams
    cudaStream_t streamA, streamB;
    cudaStreamCreate(&streamA);
    cudaStreamCreate(&streamB);

    // Create an event
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDefault); // Default behavior

    // Record an event in Stream A
    printf("Launching kernel in Stream A...\n");
    simpleKernel<<<N / 256, 256, 0, streamA>>>(d_array, 1, N);
    cudaEventRecord(event, streamA);

    // Make Stream B wait for the event in Stream A
    printf("Stream B is waiting for Stream A to complete...\n");
    cudaStreamWaitEvent(streamB, event, 0);

    // Launch a kernel in Stream B
    simpleKernel<<<N / 256, 256, 0, streamB>>>(d_array, 2, N);

    // Copy data back to host (synchronously to ensure all streams are complete)
    cudaMemcpy(h_array, d_array, SIZE, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < 10; i++) {
        printf("h_array[%d] = %d\n", i, h_array[i]);
    }

    // Cleanup
    cudaFree(d_array);
    cudaFreeHost(h_array);
    cudaStreamDestroy(streamA);
    cudaStreamDestroy(streamB);
    cudaEventDestroy(event);

    return 0;
}
```

---

## **Explanation of the Example**

1. **Stream A Operations**:
   - Launches a kernel that assigns the value `1` to the device array.
   - Records an event after the kernel execution.

2. **Stream B Dependency**:
   - Stream B is configured to wait for the event recorded in Stream A using `cudaStreamWaitEvent`.
   - After the event is completed, Stream B launches a kernel that overwrites the device array with the value `2`.

3. **Synchronization**:
   - The `cudaMemcpy` operation is performed synchronously to ensure all stream operations are complete.

---

## **Expected Output**
```
Launching kernel in Stream A...
Stream B is waiting for Stream A to complete...
h_array[0] = 2
h_array[1] = 2
h_array[2] = 2
h_array[3] = 2
h_array[4] = 2
h_array[5] = 2
h_array[6] = 2
h_array[7] = 2
h_array[8] = 2
h_array[9] = 2
```

---

## **Key Points**

1. **Inter-Stream Coordination**:
   - Stream B does not start its kernel until Stream A completes its operations and the event is satisfied.

2. **Using `cudaEventRecord` and `cudaStreamWaitEvent`**:
   - These functions allow fine-grained control over stream dependencies.

3. **Default Event Behavior**:
   - The event records timing information and synchronizes Stream B with Stream A.

4. **Concurrency**:
   - Although Stream B waits for Stream A, operations within each stream execute concurrently.

---

## **Best Practices**

1. **Use Non-Timing Events for Synchronization**:
   - If timing is unnecessary, use `cudaEventCreateWithFlags(&event, cudaEventDisableTiming)` to reduce overhead.

2. **Minimize Dependencies**:
   - Avoid excessive inter-stream dependencies, as they can reduce concurrency and GPU utilization.

3. **Leverage Dependencies for Complex Workflows**:
   - Use events strategically to coordinate tasks in multi-stream applications.

-----

# **Study Notes: Implementing Inter-Stream Dependencies with `cudaStreamWaitEvent`**

---

### **Goal**
To implement a CUDA program where:
- **Stream 1** executes a kernel while **Stream 2** loads memory to the device.
- **Stream 2** executes a kernel while **Stream 1** transfers memory back to the host.
- This alternating pattern continues with synchronization between streams.

---

### **Key Concept: Inter-Stream Dependencies**
Inter-stream dependencies allow streams to coordinate their operations by introducing a dependency between them. This is achieved using:
1. **CUDA Events**: Mark specific points in a stream’s execution.
2. **`cudaStreamWaitEvent`**: Makes one stream wait for an event recorded in another stream.

---

### **Steps to Implement Alternating Operations with Dependencies**

#### **1. Initialize Streams and Events**
- Create two CUDA streams (`stream1` and `stream2`) for the alternating operations.
- Create two CUDA events (`event1` and `event2`) to manage synchronization between streams.

```cpp
cudaStream_t stream1, stream2;
cudaEvent_t event1, event2;

cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

cudaEventCreateWithFlags(&event1, cudaEventDefault); // Default event for Stream 1
cudaEventCreateWithFlags(&event2, cudaEventDefault); // Default event for Stream 2
```

---

#### **2. Workflow Overview**

The operations alternate as follows:
1. **Stream 1**:
   - Load memory to the device.
   - Execute a kernel.
   - Transfer results back to the host.
   - Record an event (`event1`) after kernel execution.

2. **Stream 2**:
   - Wait for `event1` (to ensure Stream 1's kernel has started).
   - Load memory to the device.
   - Execute a kernel.
   - Transfer results back to the host.
   - Record an event (`event2`) after kernel execution.

3. **Stream 1**:
   - Wait for `event2` before starting its next operation.

---

#### **3. Alternating Workflow Implementation**

```cpp
for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
    // Stream 1: Load memory, execute kernel, and transfer results back
    cudaMemcpyAsync(d_data1, h_data1, SIZE, cudaMemcpyHostToDevice, stream1);
    computeKernel<<<grid, block, 0, stream1>>>(d_data1, N, 1.0f);
    cudaMemcpyAsync(h_data1, d_data1, SIZE, cudaMemcpyDeviceToHost, stream1);

    // Record an event in Stream 1
    cudaEventRecord(event1, stream1);

    // Stream 2: Wait for Stream 1, then load memory, execute kernel, and transfer results back
    cudaStreamWaitEvent(stream2, event1, 0); // Stream 2 waits for Stream 1's event
    cudaMemcpyAsync(d_data2, h_data2, SIZE, cudaMemcpyHostToDevice, stream2);
    computeKernel<<<grid, block, 0, stream2>>>(d_data2, N, 2.0f);
    cudaMemcpyAsync(h_data2, d_data2, SIZE, cudaMemcpyDeviceToHost, stream2);

    // Record an event in Stream 2
    cudaEventRecord(event2, stream2);

    // Stream 1 waits for Stream 2 before the next iteration
    cudaStreamWaitEvent(stream1, event2, 0);
}
```

---

### **How to Use `cudaStreamWaitEvent`**

#### **Function Signature**
```cpp
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
```

#### **Parameters**
1. **`stream`**:
   - The stream that will wait for the event.
   - Example: `stream2` in `cudaStreamWaitEvent(stream2, event1, 0);` makes Stream 2 wait for `event1`.
   
2. **`event`**:
   - The event that the stream waits for.
   - Example: `event1` marks a synchronization point in Stream 1.

3. **`flags`**:
   - Modifier flags for behavior customization.
   - Currently, only `0` is valid, indicating default behavior.

---

#### **Usage Example**

1. **Record an Event in Stream 1**:
   ```cpp
   cudaEventRecord(event1, stream1);
   ```

2. **Make Stream 2 Wait for Event 1**:
   ```cpp
   cudaStreamWaitEvent(stream2, event1, 0);
   ```

3. **Stream 2 Proceeds After Event 1**:
   - Once all operations in Stream 1 prior to `event1` are complete, Stream 2 continues its execution.

---

### **Diagram of Workflow**

```
Time --->

Stream 1: [Load Data] ---> [Kernel Execution] ---> [Transfer Back] ---> [Event 1 Recorded]
                                   |
                                   | Stream 2 waits for Event 1
                                   v
Stream 2:       [Load Data] ---> [Kernel Execution] ---> [Transfer Back] ---> [Event 2 Recorded]
                                   |
                                   | Stream 1 waits for Event 2
                                   v
Stream 1: [Next Load Data] ---> ...
```

---

### **Key Points to Remember**

1. **Why Use Events?**
   - Events allow precise synchronization between streams without introducing unnecessary global synchronization (like `cudaDeviceSynchronize`).

2. **Alternating Workflow Benefits**:
   - Efficiently utilizes the GPU by overlapping memory transfers and kernel execution.
   - Avoids idle time in streams, maximizing performance.

3. **Default Behavior of `cudaStreamWaitEvent`**:
   - Passing `0` as the flag ensures standard waiting behavior.

4. **Scalability**:
   - This technique can be extended to more streams or more complex workflows.

---

### **Best Practices**

1. **Use Events Strategically**:
   - Record events only where necessary to minimize overhead.
   - Use `cudaEventDisableTiming` if timing is not required.

2. **Overlap Operations**:
   - Design your workflow to maximize overlapping between streams.

3. **Avoid Excessive Dependencies**:
   - Overusing `cudaStreamWaitEvent` may serialize execution unnecessarily, reducing concurrency.

---

This structured note explains the alternating workflow implementation and usage of `cudaStreamWaitEvent` in an easy-to-understand manner. Let me know if further clarification or additional examples are needed!

-----

# **CUDA Concepts and Techniques Summary**

---

## **1. Overlapping Memory Transfer and Kernel Execution**

### Key Points:
- CUDA allows memory transfers and kernel execution to overlap using **streams**.
- By dividing work into multiple streams, you can improve GPU utilization and reduce overall execution time.

### How It Works:
- **Streams** enable concurrent operations. For example:
  - One stream transfers data while another executes a kernel.
- **Asynchronous APIs** like `cudaMemcpyAsync` and `cudaStreamSynchronize` are critical for enabling overlap.

### Practical Benefits:
- Keeps the GPU busy with computation while data transfers occur in parallel.
- Enhances performance in applications with large datasets.

---

## **2. Stream Synchronization and Blocking Behavior**

### Blocking Behavior of the NULL Stream:
- The **NULL stream** synchronizes with all blocking streams in the same context.
- Tasks in blocking streams wait for NULL stream operations to complete before proceeding.

### Types of Streams:
1. **Blocking Streams**:
   - Default behavior for streams created with `cudaStreamCreate()`.
   - Synchronizes with the NULL stream.
2. **Non-Blocking Streams**:
   - Created with `cudaStreamCreateWithFlags(cudaStreamNonBlocking)`.
   - Operate independently of the NULL stream.

### APIs for Synchronization:
- `cudaStreamSynchronize`: Blocks the host thread until all tasks in the stream are complete.
- `cudaStreamWaitEvent`: Creates inter-stream dependencies by waiting for an event in another stream.

---

## **3. Explicit and Implicit Synchronization**

### Explicit Synchronization:
- Synchronization points are explicitly defined by the programmer.
- Key APIs:
  1. `cudaDeviceSynchronize`: Waits for all device tasks to complete.
  2. `cudaStreamSynchronize`: Waits for all tasks in a specific stream.
  3. `cudaEventSynchronize`: Waits for a specific event in a stream.

### Implicit Synchronization:
- Happens automatically as a side effect of certain CUDA operations, such as:
  - Memory transfers with `cudaMemcpy`.
  - Memory allocation (`cudaMalloc`, `cudaFree`).
  - Operations in the NULL stream.
  - Switching configurations (e.g., L1 cache/shared memory).

### Practical Insight:
- Explicit synchronization offers control but can hurt performance if overused.
- Implicit synchronization is unavoidable but should be understood to avoid bottlenecks.

---

## **4. CUDA Events and Timing**

### What Are CUDA Events?
- Markers used to track progress, create dependencies, and measure performance.

### Key APIs:
1. **Event Creation**:
   - `cudaEventCreate`: Creates an event with default properties.
   - `cudaEventCreateWithFlags`: Allows customization (e.g., non-blocking or no timing).
2. **Event Recording**:
   - `cudaEventRecord`: Records an event in a specific stream.
3. **Event Synchronization**:
   - `cudaEventSynchronize`: Blocks the host until the event is complete.
   - `cudaEventQuery`: Non-blocking check for event completion.
4. **Timing**:
   - `cudaEventElapsedTime`: Measures the elapsed time between two events.

### Use Cases:
- Measure kernel execution time.
- Monitor progress within streams.
- Synchronize operations across streams.

---

## **5. Creating Inter-Stream Dependencies with Events**

### Key Concept:
- Use events to block tasks in one stream until tasks in another stream are complete.

### Workflow:
1. Record an event in **Stream A** using `cudaEventRecord`.
2. Make **Stream B** wait for the event using `cudaStreamWaitEvent`.

### Practical Example:
- Use inter-stream dependencies to control execution order in multi-stream workflows, ensuring correctness while maintaining concurrency.

---

## **6. Best Practices for CUDA Programming**

1. **Use Streams and Asynchronous APIs**:
   - Leverage streams to overlap memory transfers and computation.
   - Use `cudaMemcpyAsync` for non-blocking data transfers.

2. **Minimize Synchronization Overhead**:
   - Avoid unnecessary `cudaDeviceSynchronize` or `cudaStreamSynchronize` calls.
   - Use non-blocking streams to improve concurrency.

3. **Use Events Strategically**:
   - Record events to create dependencies or measure execution time.
   - Use `cudaEventDisableTiming` for pure synchronization to reduce overhead.

4. **Optimize Memory Usage**:
   - Use pinned memory (`cudaMallocHost`) for faster host-device data transfers.
   - Avoid excessive memory allocations, as they introduce implicit synchronization.

5. **Debug with Synchronization**:
   - Use `cudaEventQuery` or `cudaStreamSynchronize` to debug execution order without affecting performance.

---

## **7. Practical Examples**
### Topics Demonstrated:
1. Overlapping memory transfers and kernel execution with streams.
2. Blocking behavior of the NULL stream versus non-blocking streams.
3. Creating inter-stream dependencies using events.
4. Measuring execution time with CUDA events.

---

### Summary Table of Key APIs

| **Category**         | **API**                                | **Purpose**                                                                                 |
|-----------------------|----------------------------------------|---------------------------------------------------------------------------------------------|
| Stream Management     | `cudaStreamCreate`, `cudaStreamDestroy`| Create and destroy streams.                                                                |
| Stream Synchronization| `cudaStreamSynchronize`               | Wait for all tasks in a stream to complete.                                                |
| Event Management      | `cudaEventCreate`, `cudaEventDestroy`  | Create and destroy events.                                                                 |
| Event Recording       | `cudaEventRecord`                     | Record an event at a specific point in a stream.                                           |
| Event Synchronization | `cudaEventSynchronize`                | Wait for an event to complete.                                                             |
| Event Timing          | `cudaEventElapsedTime`                | Measure elapsed time between two events.                                                   |
| Inter-Stream Control  | `cudaStreamWaitEvent`                 | Make one stream wait for an event in another stream.                                       |

---

### **Final Takeaway**
The concepts and techniques you learned today provide the foundation for writing high-performance CUDA programs. By leveraging streams, events, and proper synchronization, you can maximize GPU utilization, improve execution efficiency, and handle complex workflows with ease. Let me know if you'd like additional clarification or more advanced examples!
