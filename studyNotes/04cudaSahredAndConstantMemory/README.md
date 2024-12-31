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

2. **Dynamic Shared Memory**:  
   - Size is specified during kernel launch using the third kernel parameter (e.g., `<<<blocks, threads, shared_mem_size>>>`).  
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