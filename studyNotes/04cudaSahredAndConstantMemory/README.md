# **Introduction to CUDA Shared Memory**

In CUDA programming, memory access patterns significantly impact performance. Depending on the nature of the algorithm, **misaligned** and **non-coalesced** memory accesses might be unavoidable. In such cases, **shared memory** can be used to optimize kernel performance and reduce latency.

## **Types of Memory in CUDA**
CUDA categorizes memory into two main types based on their location and latency:

### **1. On-Board Memory (High Latency)**  
   - **L2 Cache**: Moderates access between the GPU cores and global memory.  
   - **Global Memory**:  
     - High latency (~400–800 cycles) but large capacity.  
     - Accessed by all threads across all blocks.  

### **2. On-Chip Memory (Low Latency)**  
   - **Shared Memory**:  
     - On-chip and very close to streaming multiprocessors (SMs).  
     - Significantly lower latency compared to global memory.  
     - Capacity is **limited** (depends on GPU microarchitecture).  
     - **Warning:** Excessive shared memory usage per block can reduce the number of concurrently resident thread blocks per SM, thereby lowering kernel **occupancy** (percentage of active threads per SM).  

   - **L1 Cache**: Provides fast access but is not user-programmable.  
   - **Read-Only Cache**: Optimized for read-only data, typically accessed by `__restrict__` pointers.  
   - **Constant Memory**: Specialized cache for constant data shared by all threads.

---

## **Shared Memory: Common Use Cases**
Shared memory is a powerful resource for improving performance in CUDA programs. It serves several critical roles:

1. **Intra-block Thread Communication**  
   Threads within a block can use shared memory to exchange data efficiently without needing to access slower global memory.

2. **Program-managed Cache for Global Memory**  
   Shared memory can temporarily store frequently accessed global memory data, reducing latency and improving overall memory access efficiency.

3. **Scratch Pad Memory**  
   It is used to transform or reorganize data into formats that facilitate coalesced global memory accesses, improving bandwidth utilization.

---

## **Shared Memory Characteristics**
- A fixed amount of shared memory is allocated to each thread block upon its execution.  
- The shared memory is **shared by all threads within a thread block**, and its lifetime is tied to the block's execution duration.  
- Shared memory transactions are issued at the warp level:  
  - **Optimal Case:** All threads in a warp access memory addresses within the same 32-byte segment (single transaction).  
  - **Worst Case:** Memory access patterns are scattered, requiring **32 separate transactions**.

**Note:** Unlike L1 cache, which is automatic and hardware-managed, shared memory is **programmable**, allowing developers to explicitly store and reuse data.

---

## **Example: Using Shared Memory**

### **Static Shared Memory Declaration**
Static shared memory is declared at compile time using the `__shared__` keyword.

```c
__global__ void smem_static_test(int* in, int* out, int size) {
    int tid = threadIdx.x; // Thread index within block
    int gid = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

    // Declare a statically allocated shared memory array
    __shared__ int smem[SHARED_MEMORY_ARRAY_SIZE];

    // Check boundary conditions
    if (gid < size) {
        // Load data into shared memory
        smem[tid] = in[gid];
        // Perform operations and write back to global memory
        out[gid] = smem[tid];
    }
}
```

### **Dynamic Shared Memory Declaration**
Dynamic shared memory is declared at runtime using the `extern __shared__` modifier.

```c
__global__ void smem_dynamic_test(int* in, int* out, int size) {
    int tid = threadIdx.x; // Thread index within block
    int gid = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

    // Declare dynamically allocated shared memory
    // or this can be a pointer like: `extern __shared__ int* smem;`
    extern __shared__ int smem[];

    // Check boundary conditions
    if (gid < size) {
        // Load data into shared memory
        smem[tid] = in[gid];
        // Perform operations and write back to global memory
        out[gid] = smem[tid];
    }
}
```

### **Key Differences**
1. **Static Shared Memory**:  
   - Size is fixed at compile time and specified in the kernel code.  
   - Best for predictable, small-sized allocations.
   - Launch: `smem_static_test<<<grid, block>>>(d_in, d_out, size);`

2. **Dynamic Shared Memory**:  
   - Size is specified during kernel launch using the third kernel parameter (e.g., `<<<blocks, threads, shared_mem_size>>>`).
        * `smem_dynamic_test<<<grid, block, sizeof(int) * SHARED_MEMORY_SIZE>>>(d_in, d_out, size);`
   - Useful for variable or larger memory requirements.

---

## **Best Practices for Shared Memory Usage**
1. **Minimize Shared Memory Usage**  
   Over-allocating shared memory can reduce SM occupancy, degrading overall performance. Balance memory usage and active threads.

2. **Align Memory Access**  
   Ensure threads access contiguous addresses in shared memory to maximize memory bandwidth and reduce access latency.

3. **Avoid Bank Conflicts**  
   Shared memory is divided into banks. Avoid threads accessing the same bank simultaneously to prevent serialization.

4. **Combine with Registers**  
   Use registers for frequently accessed variables to further reduce memory access overhead.

5. **Leverage Dynamic Allocation Judiciously**  
   Use dynamic shared memory only when static allocation is insufficient or when sizes vary across kernel launches.

-----

# **Shared Memory Banks and Access Modes**

In CUDA, shared memory is divided into **banks** to allow parallel access by threads in a warp. Efficient utilization of shared memory depends on understanding how these banks function and avoiding conflicts that can degrade performance.

---

## **Shared Memory Banks**

Shared memory is divided into **32 equally sized memory modules**, called **banks**, which can be accessed simultaneously. Each bank serves one 32-bit or 64-bit word per cycle, depending on the GPU's compute capability.

### **Bank Layout and Word Indices**
- Banks are sequentially indexed from **0 to 31**.  
- Each bank contains a series of memory words. A **word** is a fixed-size data unit (e.g., 4 bytes for 32-bit words or 8 bytes for 64-bit words).  

#### **Example Layout: 32-bit Banks**
```
Bank Index:         0 ----- 31
=================================
4-Byte Word Index:  0 ----- 31
                    32 ---- 63
                    64 ---- 95
                    96 ---- 127
```

### **Relationship Between Banks and Warps**
Since there are **32 threads per warp**, each thread in a warp can access one bank simultaneously without conflict if the memory access pattern is properly aligned.

---

## **Bank Conflicts**

A **bank conflict** occurs when multiple threads in a warp attempt to access **different words within the same bank** in a single transaction. When this happens:
- The memory request is split into multiple **conflict-free transactions**.
- The number of transactions equals the degree of the conflict (i.e., how many threads access the same bank).  
- This reduces effective bandwidth and slows down kernel execution.

---

### **Common Access Scenarios in Shared Memory**

1. **Ideal Case: Parallel Access**
   - If all threads access **different banks**, the access occurs in a single transaction.  
   - This is the optimal pattern for shared memory usage, ensuring maximum bandwidth utilization.

2. **Sequential Access**
   - Threads access **different memory addresses** within the **same bank**.  
   - The memory accesses are serialized, degrading performance due to conflicts.

3. **Broadcast Access**
   - All threads access the **same memory address** within a single bank.  
   - The transaction is serialized but only one word is actually read. This leads to poor bandwidth utilization as many threads effectively do no useful work.

---

## **Shared Memory Bank Width and Access Modes**

The width of each bank determines how data is mapped to banks and accessed by threads.  

### **Bank Width Evolution**
- **Compute Capability 2.x GPUs**: 32-bit banks (4 bytes per word).  
- **Compute Capability 3.x and later GPUs**: 64-bit banks (8 bytes per word).  

### **Shared Memory Access Modes**
CUDA supports two primary access modes for shared memory:
1. **32-bit Access Mode:** Each thread accesses a 4-byte word.  
2. **64-bit Access Mode:** Each thread accesses an 8-byte word.  

---

## **Calculating Bank Index**

The **bank index** determines which bank a specific memory address is mapped to. For devices with 32-bit banks:
```plaintext
bank_index = (byte_address / (4 bytes per word)) % 32
```

---

## **Examples**

### **Example 1: Fermi Devices (32-bit Bank Width)**

On devices with 32-bit banks, successive 4-byte memory words are mapped to successive banks:

#### Layout:
```
Byte Address:          0 | 4 | 8 | 12 | ---- | 60 | .......
4-Byte Word Index:     0 | 1 | 2 | 3  | ---- | 15 | .......
Bank Index:            0 | 1 | 2 | 3  | ---- | 31 | .......
```

#### Memory Mapping:
```
Bank Index:         0 ----- 31
=================================
4-Byte Word Index:  0 ----- 31
                    32 ---- 63
                    64 ---- 95
                    96 ---- 127
```

---

### **Example 2: 64-bit Bank Width, 32-bit Access Mode**

On devices with 64-bit banks, successive **4-byte memory words** are mapped to **successive banks**. Each bank serves two 4-byte words:

#### Layout:
```
|  B1  |  B2  |  B3  |  B4  |  B5  | .............  |  B31  |
| 0,32 | 1,33 | 2,34 | 3,35 | 4,36 | .............  | 31,63 |
```

---

### **Example 3: 64-bit Bank Width, 64-bit Access Mode**

For 64-bit access, successive **8-byte memory words** are mapped to successive banks. Each bank serves one 8-byte word:

#### Layout:
```
|  B1  |  B2  |  B3  |  B4  |  B5  | .............  |  B31  |
| 0,2  | 3,4  | 5,6  | 7,8  | 9,10 | .............  | 62,63 |
```

---

## **Best Practices for Avoiding Bank Conflicts**

1. **Align Access Patterns**  
   Ensure threads in a warp access sequential addresses to map to different banks. This avoids conflicts and maximizes bandwidth.

2. **Pad Shared Memory Arrays**  
   Add padding to shared memory arrays to disrupt conflict-prone patterns. For example:
   ```c
   __shared__ int smem[32 + 1];  // Extra padding avoids conflicts
   ```

3. **Minimize Shared Memory Accesses**  
   Use shared memory judiciously and combine it with registers for frequently accessed data.

4. **Understand Device Architecture**  
   Be aware of the GPU's compute capability and adjust shared memory access modes accordingly.

-----

# **Row Major vs. Column Major Accesses in Shared Memory**

When working with CUDA shared memory, the access pattern to a 2D grid (or matrix) significantly affects performance due to the organization of memory banks. Understanding row-major and column-major access patterns is crucial to avoid bank conflicts and optimize memory throughput.

---

## **Matrix Organization and Initialization**

Consider a **32x32 matrix** initialized in **row-major order**, where:
- **Row-major order**: Elements are stored sequentially row by row.
- Starting from `0`, each element's value increments by `1`.

### **Matrix Layout in Shared Memory**
Assume a **32-bit bank width** (each bank stores 4-byte words). The matrix is laid out in shared memory as follows:

#### Memory Banks (32 Banks Total):
```
|  B1  |  B2  |  B3  |  B4  |  B5  | ............ |  B32  |
   0      1       2      3      4                     31
   .....................................................
   .....................................................
   .....................................................
   992    993    994    995    996                   1023
```

- **Rows** are stored contiguously in memory.  
- Each **row** is spread across all 32 memory banks.  
- The first 32 elements map to the first bank cycle, the next 32 elements to the next cycle, and so on.

---

## **Access Patterns: Row-Major vs. Column-Major**

### **1. Row-Major Access**
When accessing the matrix row by row:
- **Each thread in a warp** accesses a unique element in the same row.  
- These elements are mapped to **different memory banks**.  
- This ensures **no bank conflicts**, and the access is serviced in a **single memory transaction**.

#### Example: Row-Major Access by Warp
For **warp 0** accessing the first row:
```
Thread 0 -> Element 0 (Bank 0)
Thread 1 -> Element 1 (Bank 1)
Thread 2 -> Element 2 (Bank 2)
...
Thread 31 -> Element 31 (Bank 31)
```
- Result: Fully parallel access with optimal bandwidth utilization.

---

### **2. Column-Major Access**
When accessing the matrix column by column:
- **Each thread in a warp** accesses elements from the same column across multiple rows.  
- All these elements are mapped to the **same memory bank**.  
- This causes a **bank conflict**, as all threads attempt to access the same bank simultaneously.  
- CUDA hardware resolves this by serializing the requests, requiring **32 separate transactions per warp**.

#### Example: Column-Major Access by Warp
For **warp 0** accessing the first column:
```
Thread 0 -> Element 0 (Bank 0)
Thread 1 -> Element 32 (Bank 0)
Thread 2 -> Element 64 (Bank 0)
...
Thread 31 -> Element 992 (Bank 0)
```
- Result: **32 bank conflicts**, reducing bandwidth utilization and increasing latency.

---

## **Impact of Access Patterns on Performance**

1. **Row-Major Access:**
   - Accesses are aligned with the shared memory bank structure.
   - Single memory transaction per warp.
   - Ideal for memory-intensive operations where performance is critical.

2. **Column-Major Access:**
   - Causes severe bank conflicts, reducing performance.
   - Each warp requires multiple serialized transactions, increasing kernel execution time.

---

## **How to Avoid Bank Conflicts in Column-Major Access**

If column-major access is unavoidable, **padding** can be used to disrupt conflict patterns:
- Introduce an extra column of padding between rows in shared memory.  
- This shifts the bank index of each column, reducing conflicts.

### **Example: Padding to Avoid Conflicts**
For a 32x32 matrix:
```c
__shared__ int smem[32][33];  // Add 1 column of padding
```
- The extra column ensures that successive rows are no longer mapped to the same banks.

---

## **Key Takeaways**
1. **Understand Memory Layout**: Row-major storage aligns with CUDA’s shared memory bank structure, while column-major storage often leads to conflicts.
2. **Align Access Patterns**: Ensure threads access memory in ways that minimize conflicts.
3. **Use Padding**: Add padding to disrupt conflict-prone access patterns in column-major layouts.

-----

# Static and Dynamic Shared Memory

Use `index = threadIdx.y * blockDim.x + threadIdx.x` to access in row major format.

See staticDynamicDemo.cu