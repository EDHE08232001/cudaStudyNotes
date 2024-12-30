# Threads
A thread is purely a logical concept and represents a sequence of instructions to be executed:
  - it does not exist as hardware but is mapped onto the hardware (like CUDA cores) during execution

## Physical Component Interaction
Threads are executed on physical hardware units, such as CUDA cores in a Streaming Multiprocessor. Each CUDA core can process instructions for one thread at a time, but because GPUs can interleave threads, thousands of threads can appear to run concurrently.

---

# Registers
Registers in a GPU are **high-speed, small-sized memory locations** directly accessible by processing unit (cuda core). They are used for storing variables or intermediate results during computation.
  - Private registers for each thread
    - private to the thread: no other thread can access these registers
    - used to store local variables or temporary results that are specific to the thread
    - allocated automatically by the compiler based on the thread's code

## Why are register private to threads?
  - Performance: quick access to each thread's variables for a specific thread
  - Isolation
  - Efficiency
    - Register usage helps avoid slower memory accesses

-----

# CUDA Execution Model

## 1. GPU Architecture Overview
The GPU architecture is built around a scalable array of **Streaming Multiprocessors** (SMs). GPU hardware parallelism is achieved through replication of this architectural building block. This means that each SM handles multiple threads concurrently, which allows GPUs to excel in parallel computations.

### Explanation
- **Streaming Multiprocessors (SMs)** are the core processing units in the GPU, enabling efficient parallelism.
- GPUs achieve parallelism by using many SMs working in parallel, each capable of managing thousands of threads simultaneously.
- Each SM contains several subcomponents that facilitate parallel execution, like:
  - CUDA Cores
    - Primary processing units within an SM
    - Analogous to cores in a CPU
    - Can perform arithemetic and logical operations for threads
  - Warp Scheduler
    - Manages execution of multiple warps(groups of 32 threads) within the SM
    - Issues instructions to the CUDA cores in a SIMT manner
  - Registers
    - Each thread running on the GPU has its own private registers
    - These are used for storing **thread-specific** variables and intermediate computation results
  - Shared Memory
    - **User-Manages** memory space that is **shared** among all threads in a block
    - has lower latency than global memory
  - Instruction Units
  - Specialized Units
    - Tensor Cores
      - designed for matrix operation in ML and Deep Learning applications
      - Texture Units
  - L1 Caches
    - each SM includes an L1 cache to store frequently used data and reduce global memory access latency
  - Other Components

## 2. Key Components of Streaming Multiprocessors (Fermi Micro-architecture)

Using the **Fermi micro-architecture** as an example, the key components of SMs include:

1. **CUDA Cores**: Execute instructions in parallel.
2. **Shared Memory and L1 Cache**: Serve as fast, on-chip memory accessible by threads within the same block.
3. **Registers**: Store data for active threads; their capacity is limited, so careful management is essential.
4. **L/S Units (Load and Store Units)**: Handle memory operations, moving data between memory and registers.
5. **Warp Schedulers**: Schedule instructions for execution across CUDA cores.
6. **SFU (Special Function Units)**: Perform specialized calculations, such as trigonometric and logarithmic operations.

**Note**: These components are on-chip, meaning resources like registers, shared memory, and L1 cache are limited. Effective resource management is critical. Warp schedulers consider resource usage when determining which warp will execute next.

### Explanation
- Fermi is an older architecture, but understanding its core components provides foundational knowledge.
- Modern SMs are more complex, with additional optimizations and components.

## 3. Modern SM Example: Volta Micro-architecture

The **Volta micro-architecture** introduces more specialized cores:
- **Tensor Cores**: Accelerate matrix multiplications, especially useful in deep learning applications.
- **Integer Operation Cores**: Handle integer calculations.
- **FP32 Cores**: Optimize 32-bit floating-point operations.
- **FP64 Cores**: Optimize 64-bit floating-point operations.

### Explanation
- Volta’s architecture highlights GPU evolution towards more specialized, efficient computation, especially for AI and scientific applications.
- Tensor cores, for example, provide dramatic performance boosts in neural network training and inference.

## 4. Flynn's Taxonomy of Computer Architectures

Flynn's Taxonomy classifies computer architectures into four categories:

1. **SISD (Single Instruction, Single Data)**
   - Represents sequential computers that do not exploit parallelism.
   - **Text Visualization**: `Instruction Pool -----> PU <-------- Data Pool`

2. **SIMD (Single Instruction, Multiple Data)**
   - Contains a control unit, processing unit, and memory unit.
   - Instructions execute sequentially but can process multiple data points simultaneously (achieved via pipelining or multiple function units).

3. **MISD (Multiple Instructions, Single Data)**
   - Executes multiple instructions on a single data stream.
   - This architecture is uncommon and mainly used for fault-tolerant systems.

4. **MIMD (Multiple Instructions, Multiple Data)**
   - Multiple autonomous processors execute different instructions on different data simultaneously.
   - Examples include multi-core processors and distributed systems with shared or distributed memory spaces.

## 5. CUDA Architecture: SIMT (Single Instruction, Multiple Threads)

CUDA follows a unique version of computer architecture called **SIMT**:
- **SIMT**: Single instruction runs on multiple threads, adhering to GPU parallelism principles.

### CUDA Execution Model
1. **Thread Blocks**: Execute within a single SM. Multiple thread blocks can execute simultaneously on the same SM, depending on resource limits.
   - A single thread block cannot execute across multiple SMs.
   - If a single block cannot fit into one SM, an error will occur at kernel launch.

2. **Software Level Grid Organization**
   - Threads are organized into blocks, which form a grid.
   - The grid is mapped to the device at the hardware level.
   - Thread blocks in the grid are distributed across SMs on the device.

3. **Thread Execution**
   - A single thread within a block is executed by a single core.
   - Since there are multiple cores in an SM, the same instruction for multiple threads can execute in parallel, adhering to SIMT principles.

### Explanation
- SIMT allows GPUs to achieve high throughput by executing the same instruction across multiple threads simultaneously.
- Proper thread organization and management are crucial for optimal performance.

## 6. Considerations for CUDA Programming
When programming with CUDA, it's important to be aware of which device is being used, as different CUDA microarchitectures have distinct capabilities:
- Some microarchitectures offer more cores, memory bandwidth, or special units like tensor cores, impacting performance.
- Always check device properties (e.g., number of SMs, available memory) to ensure code is optimized for the target hardware.

### Example Code
```cpp
// Example: Launching a CUDA Kernel
__global__ void add(int *a, int *b, int *c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}

int main() {
    // Code to allocate memory, copy data, and launch kernel goes here.
}
```
- In this code, the grid and block configuration needs to be defined properly based on the device's capacity.

-----

# WAPRS (Warp Scheduling in CUDA)

## 1. Logical vs. Hardware Parallelism

When launching CUDA kernels, execution may appear parallel logically, as multiple threads run concurrently. However, actual execution differs from a hardware perspective due to the architecture of the GPU.

### Explanation
- **Logical Parallelism**: This describes how the code is designed to run concurrently, with many threads launched at once.
- **Hardware Parallelism**: This focuses on how the GPU executes threads based on its hardware capabilities, such as the number of Streaming Multiprocessors (SMs) and cores.

## 2. Warps: The Basic Unit of Execution

CUDA divides **thread blocks** into smaller units called **warps**, each consisting of 32 consecutive threads. Warps are the fundamental unit of execution for the GPU.

### Key Points
- **Warp Size**: Always 32 threads.
- **Warps per Block**: Calculated as:
  * Calculated as Block Size divided by 32
- All warps within a given thread block are executed on the same SM.

### Example: Warp Calculation and Execution

Consider the following scenario:
- **Total threads**: 1 million (1M)
- **Threads per block**: 512

Device:
- **Number of SMs**: 13
- **Cores(Threads) per SM**: 128

#### Detailed Calculation
1. **Calculate Warps per Block**:
  * 512 threads per block divided by 32 gives 16 warps
2. **Total Number of Cores**:
   - 13 SMs multiplied by 128 cores per SM gives 1,664 cores
   - So this device can only execute 1664 threads parallel
3. **Warp Execution**:
   - Each SM can handle 128 threads concurrently, corresponding to 4 warps (128/32 = 4).
   - The execution of warps depends on factors such as memory readiness and resource availability.

### Explanation
- **Warps are the basic unit of execution in an SM**. Once a thread block is scheduled on an SM, it is divided into warps, with each warp executing instructions in a **SIMT (Single Instruction, Multiple Threads)** manner, meaning all threads in the warp run the same instruction simultaneously.

## 3. Thread Block Configuration

CUDA allows 1D, 2D, and 3D thread block configurations, each having specific execution behavior.

### 1D, 2D, and 3D Configurations
- **1D Block**: Threads are arranged in a single dimension.
- **2D/3D Blocks**: Threads are structured in two or three dimensions but treated as 1D by the hardware, with each thread assigned a unique ID.

#### Example 1: 1D Block Configuration
- **Block Size**: 128 threads
  - Divided into 4 warps:
    - **Warp 0**: Threads 0 to 31
    - **Warp 1**: Threads 32 to 63
    - **Warp 2**: Threads 64 to 95
    - **Warp 3**: Threads 96 to 127

#### Example 2: 2D Block Configuration
- **Block Size**: 64 threads in the x-dimension and 2 threads in the y-dimension (total 128 threads)
  - Similarly divided into 4 warps, as in the 1D case.

#### Example 3: 2D Block with 80 Threads (40x2)
- 80 threads are divided into 3 warps.
  - **Warp Calculation**: \(80/32 = 2.5\), rounded up to 3 warps.
  - Warp allocation:
    - **Warp 0**: Threads 0 to 31
    - **Warp 1**: Threads 32 to 39 (remaining threads inactive)
    - **Warp 2**: Threads 40 to 71
    - **Warp 3**: Threads 72 to 79 (remaining threads inactive)
- **Remember:** All threads in a warp should be from the same block

### Explanation
- **Inactive Threads**: CUDA always allocates complete warps, so some threads may be inactive.
  - Inactive threads still consume resources, such as shared memory and registers, which can contribute to inefficiency.

## 4. Calculating Warp Index

CUDA does not provide a built-in variable for the **warp index**. However, it can be calculated manually:
- WaprIndex = threadIdx.x / 32
- This formula assigns an index to each warp within a block, with each set of 32 consecutive threads having a unique warp index.
- The same warp index can exist across different blocks.

### Example
- **Block Size**: 512 threads
  - Warp indices:
    - **Warp 0**: Threads 0 to 31
    - **Warp 1**: Threads 32 to 63
    - ...
    - **Warp 15**: Threads 480 to 511

### Explanation
- Each warp index groups 32 consecutive threads.
- Across multiple blocks, the same warp index can reoccur, but each is independently scheduled.

## 5. Impact of Inactive Threads

Inactive threads in a warp still consume resources like registers and shared memory. For example, if a block has only one active thread, it still allocates a full warp, resulting in 31 inactive threads.

### Best Practices
To minimize memory wastage:
- **Configure block sizes as multiples of 32** in the x-dimension to ensure all threads in each warp are active.

## 6. Example Code: 1D Grid with Proper Block Size
```cpp
// Example: Launching a CUDA Kernel with optimized block size
__global__ void add(int *a, int *b, int *c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int warpIdx = threadIdx.x / 32; // Calculate warp index
    c[index] = a[index] + b[index];
}

int main() {
    int numThreads = 1024;  // Multiple of 32
    int blockSize = 256;    // Multiple of 32
    int gridSize = (numThreads + blockSize - 1) / blockSize;

    // Allocate memory, initialize data, and launch kernel
    add<<<gridSize, blockSize>>>(a, b, c);
}
```

### Explanation
- The above code calculates the warp index within each block.
- Both the total number of threads and block size are set as multiples of 32, ensuring all threads are active.

## Summary
- **Warps** are the basic unit of execution in CUDA, consisting of 32 threads.
- Efficient CUDA programming requires managing warps effectively to optimize resource usage.
- Configure block sizes as multiples of 32 to minimize inactive threads and improve memory utilization.

```cpp
#include <stdio.h>

__global__ void print_details_of_warps() {
    // Global thread ID calculation
    int gid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    // Warp ID within the block
    int warp_id = threadIdx.x / 32;

    // Global block ID calculation (flattened 2D grid index)
    int gbid = blockIdx.y * gridDim.x + blockIdx.x;

    // Print thread details
    printf(
        "Thread ID: %d, Block ID (x): %d, Block ID (y): %d, Global Thread ID: %d, Warp ID: %d, Global Block ID: %d\n",
        threadIdx.x,
        blockIdx.x,
        blockIdx.y,
        gid,
        warp_id,
        gbid
    );
}

int main() {
    // Define block size and grid size
    dim3 block_size(42);
    dim3 grid_size(2, 2); // 2x2 grid

    // Launch kernel
    print_details_of_warps<<<grid_size, block_size>>>();

    // Synchronize and reset the device
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}
```

-----

# CUDA Programming: Warp Divergence

## What is Warp Divergence?
- **Warp divergence** occurs when threads within the same warp execute different instructions.
  - Example:
    ```cpp
    if (tid < 16) {
        // if block
    } else {
        // else block
    }
    ```
- In CUDA, warps execute in **SIMT (Single Instruction Multiple Threads)** fashion, meaning all threads in a warp are expected to execute the same instruction simultaneously.

## Behavior of Divergent Warps
- When threads in a warp diverge, the warp must **serially execute each branch path**, disabling the threads that are not taking that particular path.
  - For example:
    ```cpp
    int tid = threadIdx.x;

    if (tid % 2 == 0) {
        // do something
    } else {
        // do something else
    }
    ```
  - In this scenario, CUDA ensures that each thread sequentially executes both the `if` and `else` blocks, while temporarily disabling the threads that do not satisfy the current branch condition.

## Key Points to Remember
- **Control flow statements** such as `if-else` and `switch` may indicate potential divergent code.
- However, the mere presence of a control flow statement doesn't always lead to warp divergence.
  - **Warp divergence** occurs when there are multiple paths of execution within the same warp.
  - If all threads within a warp follow the same path, there is **no divergence**.

## Optimizing Warp Execution
- Aim to create condition checks that **do not induce warp divergence**.
- Example: A condition that leads to all threads executing the same path avoids divergence.

## Measuring Warp Divergence
- To determine if your kernel contains divergent code, use performance metrics available in the **`nvprof` profiling tool**, which allows you to analyze the efficiency of branches and detect divergence.

## Calculating Branch Efficiency
- **Branch Efficiency** measures how efficiently threads in a warp follow the same execution path.
  - The formula for branch efficiency is:
    100 * [{#Branches} - {#DivergentBranches} / {#Branches}]
- **Example Calculation:**
  - Suppose a kernel has 200 branches and 50 divergent branches:
    {Branch Efficiency} = 100 * [frac{200 - 50} / {200}] = 75%

  - This means that 75% of branches are efficiently executed without divergence.

# Performance Metrics - Branch Efficiency

## Compile
  - optimized
    - `nvcc file.cu -o file.out`
  - unoptimized
    - `nvcc -G file.cu -o file.out`
  - analyze performance
    - `nvprof --metrics branch_efficiency ./file.out`

```cpp
/*
CUDA Warp Divergence Example

To use performance metrics:
- Compile(Optimized): nvcc file.cu -o file.out
- Compile(Unoptimized): nvcc -G file.cu -o file.out
- Analyze performance: nvprof --metrics branch_efficiency ./file.out
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Kernel without warp divergence
__global__ void code_without_divergent() {
    // Calculate global thread ID
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Variables to hold values for different branches
    float a, b;
    a = b = 0;

    // Calculate the warp ID
    int warp_id = gid / 32; // Each warp contains 32 threads

    // Conditional branch based on warp ID (avoids divergence)
    // Since all threads in the same warp will have the same warp_id,
    // this branch ensures all threads in a warp take the same path.
    if (warp_id % 2 == 0) {
        a = 100.0;
        b = 50.0;
    } else {
        a = 200.0;
        b = 75.0;
    }

    // Result: No warp divergence, as all threads within a warp execute the same branch.
}

// Kernel with warp divergence
__global__ void divergence_code() {
    // Calculate global thread ID
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Variables to hold values for different branches
    float a, b;
    a = b = 0;

    // Conditional branch based on thread ID (causes divergence)
    // In this case, the condition is based on gid, so threads within
    // the same warp may take different paths, causing divergence.
    if (gid % 2 == 0) {
        a = 100.0;
        b = 50.0;
    } else {
        a = 200.0;
        b = 75.0;
    }

    // Result: Warp divergence occurs here, as threads in the same warp
    // may execute different branches, leading to serialization of execution.
}

int main(int argc, char** argv) {
    printf("\n---------- CUDA WARP DIVERGENCE EXAMPLE ----------\n\n");

    // Define the total number of threads (large size for testing)
    int size = 1 << 22; // 2^22 threads

    // Define block and grid sizes
    dim3 block_size(128); // 128 threads per block
    dim3 grid_size((size + block_size.x - 1) / block_size.x); // Calculate grid size

    // Launch the kernel without warp divergence
    printf("Launching kernel without warp divergence...\n");
    code_without_divergent<<<grid_size, block_size>>>();
    cudaDeviceSynchronize(); // Wait for the kernel to complete

    // Launch the kernel with warp divergence
    printf("Launching kernel with warp divergence...\n");
    divergence_code<<<grid_size, block_size>>>();
    cudaDeviceSynchronize(); // Wait for the kernel to complete

    // Reset the CUDA device
    cudaDeviceReset();

    printf("\n---------- END OF EXAMPLE ----------\n");
    return 0;
}
```

-----

# Resource Partitioning and Latency in CUDA

In CUDA programming, efficient resource partitioning and managing latency are crucial for maximizing GPU performance. Each CUDA kernel launch needs to consider how resources like registers and shared memory are distributed and how latency can be managed through warp scheduling. Understanding these core concepts can help you optimize your CUDA applications.

### Resource Partitioning in the CUDA Architecture

The local execution context of a warp, which represents a group of 32 threads, mainly consists of the following resources:

- **Program Counters**: Tracks the execution position within a warp.
- **Registers**: Stores data temporarily for each thread, including variables and temporary calculations.
- **Shared Memory**: Provides fast access to data that is shared within a thread block.

**Note:** The execution context of each warp processed by a Streaming Multiprocessor (SM) is maintained *on-chip* throughout the warp’s lifetime, which means that switching from one warp’s execution context to another incurs no cost. This zero-cost switching allows the SM to quickly swap between warps, reducing idle time and improving efficiency.

#### Programmer Control Over Resources

CUDA enables programmers to manage how registers and shared memory are allocated. Each SM has a set of 32-bit registers stored in a register file and a fixed amount of shared memory that are divided among thread blocks.

The following factors influence the number of thread blocks and warps that can simultaneously reside on an SM for a given kernel:
- **Register Usage**: Each SM has a limited number of registers; excessive use per thread reduces the number of warps that can fit on an SM.
- **Shared Memory Usage**: Shared memory is also limited per SM, and its allocation impacts how many blocks can reside on an SM.

Balancing resource usage (fewer blocks with more shared memory vs. more blocks with less shared memory) is key to optimizing performance.

---

## Warp Categories in an SM

Warps within an SM fall into different categories based on their execution status:

1. **Active Warps**: Warps within a block that have been allocated compute resources like registers and shared memory. These warps are subdivided as:
   - **Selected Warps**: Actively executing instructions.
   - **Stalled Warps**: Temporarily unable to execute, often due to waiting for data or resources.
   - **Eligible Warps**: Ready to execute but not currently being executed.

The SM’s **warp scheduler** selects warps from active warps each execution cycle and dispatches them to the execution units.

### Eligibility Conditions for Warp Execution

To be eligible for execution, a warp must meet the following conditions:
   - **Available CUDA Cores**: 32 CUDA cores should be free for execution.
   - **Instruction Ready**: All arguments for the current instruction must be available.

CUDA architectures (compute capability 3.0 and above) can support up to 64 concurrent warps on each SM. When a warp stalls, the warp scheduler will replace it with an eligible warp to keep execution flowing smoothly. Since the compute resources are kept on-chip, switching contexts incurs no performance penalty.

---

## Understanding Latency

Latency in CUDA refers to the number of clock cycles between an instruction being issued and its completion. Latency varies depending on the type of operation:

### Arithmetic Instruction Latency
For arithmetic operations, when an instruction is issued to the ALU (Arithmetic and Logic Unit), it may take several clock cycles to complete. This latency depends on the complexity of the arithmetic operation.

### Memory Operation Latency
In memory operations, latency is the number of clock cycles between issuing a memory instruction and when the data arrives at its destination. This latency can be significant for memory-bound applications, where efficient memory access patterns are essential.

---

## Latency Hiding in CUDA

CUDA hides latency by maintaining the execution context of each warp on-chip for its entire lifetime, allowing quick switching between warps. When a warp stalls, a large number of eligible warps enables the warp scheduler to dispatch another warp to execute, thereby hiding latency.

### Ensuring Sufficient Warp Availability

Having a high number of eligible warps ensures that the warp scheduler can replace stalled warps, allowing computation to continue without interruption. The goal is to keep the GPU cores busy and minimize idle cycles.

**Practical Tip:** You can find the memory clock speed of a GPU using the command `nvidia-smi -a -q -d CLOCK`.

---

## Categorizing CUDA Applications

CUDA applications generally fall into two performance categories:

- **Bandwidth-bound applications**: Dominated by memory latency. Performance is limited by the memory bandwidth, where faster data transfers are critical.
- **Compute-bound applications**: Dominated by arithmetic operations. Performance depends on the efficiency of computational resources, with a focus on minimizing execution time for complex calculations.

-----

# Arithmetic Latency and Warp Scheduling on an SM

Imagine a CUDA program where each warp is executing floating-point divisions on an NVIDIA GPU with a high core count. Floating-point division is a relatively slow arithmetic operation and may require 20–30 clock cycles to complete, creating **arithmetic latency**. Let’s break down how the SM manages this:

1. **Context Switching Within the SM**: On an SM, multiple warps are loaded and ready to execute. Suppose Warp A is performing a division operation across its threads. During this period, Warp A cannot proceed to the next instruction until the division completes.
   
2. **Scheduler Action**: The SM has a **warp scheduler** that identifies when a warp is stalled (e.g., due to arithmetic latency). When Warp A stalls, the scheduler checks for other **eligible warps** that are ready to execute, such as Warp B or Warp C.

3. **Context Switching to Hide Latency**: The scheduler rapidly switches to Warp B or C, moving them to the execution units while Warp A waits for its division result. Since each warp’s context is stored on-chip, the context switching happens with negligible overhead. This switching hides the arithmetic latency of Warp A, keeping the SM cores busy with other work.

4. **Return to Warp A**: Once Warp A's division operation is complete, the scheduler can return to Warp A, allowing it to continue with the next instruction. This rapid warp switching is essential for minimizing idle time in the SM and maintaining high utilization.

**Example Impact**: In a device with compute capability 7.0 (Volta architecture) or later, each SM can support up to 64 warps. By having many eligible warps, the device can hide arithmetic latency through frequent context switching, maximizing throughput.

---

# Memory Latency and Context Switching on an SM

Now, consider a more memory-bound operation where a warp accesses global memory, which has significantly higher latency than on-chip operations. For example, suppose Warp D is fetching a large dataset from global memory to perform a computation.

1. **High Latency from Global Memory Access**: When Warp D issues a global memory load, it may take hundreds to thousands of clock cycles for the data to arrive in the SM due to the distance and lower speed of global memory compared to on-chip memory. This **memory latency** can stall Warp D for an extended period.

2. **Scheduler Response**: The SM’s warp scheduler detects that Warp D is stalled due to memory latency. It then checks for other **eligible warps** within the SM that are ready to execute. If Warp E or Warp F is available, the scheduler quickly switches to one of these warps.

3. **Latency Hiding with Context Switching**: While Warp D waits for its memory load to complete, the SM’s scheduler keeps the GPU cores active by switching to Warp E and F. These warps execute instructions that don’t require waiting on global memory, effectively hiding Warp D’s memory latency.

4. **Data Arrival and Resume of Warp D**: Once the data for Warp D arrives in the SM, the scheduler can switch back to Warp D, allowing it to continue its work with minimal delay. This technique is effective because the scheduler dynamically manages warp execution based on their readiness, ensuring minimal idle time even with high memory latency.

**Example Impact**: On devices with compute capability 7.x, an SM can have up to 1024 resident threads, meaning several warps can be ready to execute simultaneously. For memory-bound applications, having multiple eligible warps to cover for high-latency memory operations is essential for achieving optimal performance.

---

# Summary: SM-Level Warp Scheduling and Latency Hiding

In both arithmetic and memory latency scenarios, **warp scheduling and context switching** on an SM are critical to CUDA’s ability to hide latency:

- **Arithmetic Latency**: Caused by slower, multi-cycle operations like division. The scheduler switches to another warp, keeping the SM active.
- **Memory Latency**: Caused by the long delay of fetching data from global memory. The scheduler utilizes other warps to maintain throughput while waiting for memory.

By having multiple warps in different stages of readiness, the SM’s scheduler effectively minimizes idle time and maximizes core utilization on the device. This method allows CUDA to efficiently handle both computation-bound and memory-bound workloads.

-----

# CUDA Occupancy

## Definition of Occupancy
**Occupancy** in CUDA is the ratio of active warps to the maximum number of warps per Streaming Multiprocessor (SM):

Occupancy = Active Warps / Maximum Warps

The **maximum warps** represent the upper limit of warps that can be active on each SM, determined by the device's microarchitecture.

### Purpose of High Occupancy
When a warp stalls due to latency in arithmetic or memory operations, the SM can switch to another eligible warp to maintain utilization of the cores. **Higher occupancy** indicates a sufficient number of eligible warps available to keep cores busy even when some warps experience stalls. This helps maximize performance.

## Determining Maximum Warps in an SM
The **maximum warps per SM** can be determined by:
- Reviewing the **device's microarchitecture documentation** for the specific limit, or
- **Querying the device** for the maximum number of threads per SM and dividing it by the warp size (32).

### Factors Affecting Active Warp Count
The **active warp count** per SM depends on:
- The available **hardware resources** in the device, such as registers and shared memory.
- The **resource consumption** of your kernel, specifically the registers and shared memory each block requires.

## Example Calculation of Occupancy

### Assumptions
Let's consider an example scenario to calculate occupancy:

1. **Kernel Parameters**:
   - Each thread uses **48 registers**.
   - Each block uses **4096 bytes of shared memory (Smem)**.
   - **Block size** is **128** threads.

2. **Device Properties** (NVIDIA GTX 970):
   - **65536 registers** per SM.
   - **98304 bytes of shared memory** per SM.
   - **Warp allocation granularity** of **4** (warps are allocated in groups of 4).
   - **Compute capability** of **5.2**, allowing a **maximum of 64 warps per SM**.

### Step-by-Step Calculation

#### Step 1: Determine Registers Per Warp
Each warp (32 threads) requires:

number of registers per warp = 48 * 32 = 1536 (number of registers per thread times number of threads per warp)

#### Step 2: Calculate Allowed Warps Based on Register Availability
Given that each SM has 65536 registers:

Allowed Warps = 65536 / 1536 = 42.67 (total number of registers divided by number of registers per warp)

However, due to the **warp allocation granularity** of 4, SMs can only allocate warps in multiples of 4. Therefore, we round down to the nearest multiple of 4:

Allowed Warps (based on register usage) = 40

#### Step 3: Determine Allowed Thread Blocks Based on Shared Memory
Each SM has **98304 bytes of shared memory**, and each block uses **4096 bytes** of shared memory:

Allowed Blocks (active blocks) = 98304 / 4096 = 24

Since each block is **128 threads**, this equates to **4 warps per block**:

Total Warps Based on Shared Memory = 24 * 4 = 96

**However**, for a device with **compute capability 5.2**, the maximum number of warps per SM is capped at 64.

**NOTE:** This mean our active warp count per SM is not limited by shared memory usage.

#### Step 4: Identify Limiting Resource
Since **register usage** limits the number of active warps to 40 (as opposed to 64), **registers become the limiting factor**. Thus, the active warp count per SM is limited by register usage rather than shared memory.

### Final Calculation of Occupancy
Given that the device allows a maximum of 64 warps per SM and our active warps are capped at 40 (which we calculated based on Register Availability):

Occupancy = 40/64 = 0.63 = 63%


## Summary
In this example, we calculate the occupancy as follows:
- **Register usage** limits the number of active warps per SM to 40.
- The **occupancy** achieved is **63%**, meaning the device will be able to keep 63% of its cores busy even if some warps experience stalls.

### Key Takeaways
- High occupancy helps maintain core utilization, especially if one or more warps stall.
- The **limiting factor** (registers or shared memory) depends on the resource usage of the kernel.
- Always check device properties (like max warps per SM) and granularity constraints to determine occupancy accurately.
- **Note:** Higher occupancy does not necessarily mean better performance
  - If a kernel is not bancwidth-bound or computation-bound, then increasing occupancy will not necessarily increase performance.

-----

# Occupancy Calculator
The CUDA Toolkit includes a **spreadsheet**, called the **CUDA OCCUPANCY CALCULATOR**, which assists you in selecting grid and block dimensions to maximize occupancy for a kernel.

-----

# `ptxas` options in `nvcc`
`ptxas` options in the cuda compiler `nvcc` provide detailed information about resource usage for each kernelin your cuda program. This includes:
  - registers per thread
  - shared memory per block
  - other memory usage

## What does this mean?
When compiling, `--ptxas-options=-v` flag, the `nvcc` compiler passes this option to the `PTX assembler (ptxas)`, which generates the following resource usage statistics for each kernel:
  1. registers per thread
  2. shared memory per block
  3. local memory per thread
    - provate memory used by individual threads (not to be confused with registers)
  4. constant memory usage

## How to use?
```
nvcc --ptxas-options=-v ./program.cu -o ./my_program
```

-----

# **Profiling with `nvprof`**

## **What is Profile-Driven Optimization?**

Profile-driven optimization is an iterative process that uses profiling information to identify performance bottlenecks in a program and optimize them systematically.

```
      |-----------------------|
      |                       ⬇
implementation            profiling
      ⬆-----------------------|
```

1. **Implementation:** Write or modify the CUDA program.
2. **Profiling:** Use tools like `nvprof` to analyze the performance.
3. **Optimization:** Based on profiling results, adjust the program to address bottlenecks.
4. Repeat until optimal performance is achieved.

---

### **Key Notes:**
- **Avoid Timing Kernels While Profiling:**
  - Profiling introduces a significant overhead to kernel execution. Timing kernels during profiling may yield inaccurate results.
  - Profiling is for analyzing performance metrics, not for precise time measurement.

- **`nvprof` vs CPU Comparisons:**
  - `nvprof` is not suitable for comparing GPU kernel execution times with CPU implementations.
  - It is acceptable to compare the performance of **different CUDA kernels** using `nvprof`.

---

## **`nvprof` Profiling Modes**

`nvprof` offers multiple modes to analyze different aspects of GPU performance:

1. **Summary Mode (Default):**
   - Provides a high-level summary of kernel execution and memory operations.
   - Outputs one result line per kernel function and each type of CUDA memory copy operation.

2. **GPU and API Trace Mode:**
   - Captures detailed timing information for GPU activities (e.g., kernel execution) and CUDA API calls.

3. **Event Metrics Summary Mode:**
   - Aggregates performance metrics (e.g., warp divergence, achieved occupancy) across the entire application.

4. **Event and Metrics Trace Mode:**
   - Provides detailed, per-instance data for selected events and metrics during the application’s runtime.

---

## **Compilation Tips for Accurate Profiling**

- **Compile Your Program Correctly:**
  - Avoid flags like `-g` or `-G` during compilation, as these disable compiler optimizations and can skew profiling results.

  Example compilation command:
  ```bash
  nvcc -o profiling_test.out 7_sum_array.cu
  ```

---

## **Using `nvprof`**

### **Basic Syntax:**
```bash
nvprof [options: metrics, events...] [application] [application-arguments]
```

### **Default Mode (Summary Mode):**
```bash
nvprof ./profiling_test.out
```

- Provides a summary of kernel execution time and memory copy operations.
- Useful for identifying high-level performance hotspots.

---

## **Event Metrics Summary Mode**

### **What Are Metrics?**
Metrics are measurements of GPU performance for your CUDA application. They provide insights into how efficiently your program uses GPU resources, such as:
- **SM Efficiency**: Utilization of the Streaming Multiprocessor.
- **Warp Divergence**: Degree of warp-level instruction divergence.
- **Memory Efficiency**: Efficiency of global memory loads and stores.
- **Branch Efficiency**: Efficiency of conditional branches.
- **Instruction Throughput**: Number of instructions executed per warp.

### **Common Metrics:**
- **sm_efficiency**: Measures the utilization of streaming multiprocessors.
- **achieved_occupancy**: Indicates how well the GPU is utilized by active warps.
- **branch_efficiency**: Measures how efficiently branches are resolved in the code.
- **gld_efficiency**: Efficiency of global memory loads.
- **gld_throughput**: Throughput of global memory reads.
- **dram_read_throughput**: Throughput of reads from global (device) memory.
- **inst_per_warp**: Number of instructions executed per warp.
- **stall_sync**: Measures the percentage of stalls caused by synchronization.

### **Example Commands:**

#### Collecting Specific Metrics:
```bash
nvprof --metrics gld_efficiency,sm_efficiency,achieved_occupancy ./profiling_test.out
```

#### Profiling with Application Arguments:
```bash
nvprof --metrics gld_efficiency,sm_efficiency,achieved_occupancy ./profiling_test.out 1 25 20 7 2
```

---

## **GPU and API Trace Mode**
- Use this mode to capture timing details of kernel execution and CUDA API calls.
- Example Command:
  ```bash
  nvprof --print-gpu-trace ./profiling_test.out
  ```

---

## **Example Workflow**
1. **Compile the Program:**
   ```bash
   nvcc -o profiling_test.out my_kernel.cu
   ```

2. **Run `nvprof` in Summary Mode:**
   ```bash
   nvprof ./profiling_test.out
   ```

3. **Analyze Metrics for Optimization:**
   - Example Command:
     ```bash
     nvprof --metrics sm_efficiency,gld_efficiency ./profiling_test.out
     ```

4. **Iterate and Optimize:**
   - Use profiling results to optimize kernel code, such as:
     - Reducing warp divergence.
     - Improving memory coalescing.
     - Minimizing global memory accesses with shared memory.
   - Repeat profiling to verify improvements.

---

## **Tips for Effective Profiling**
- Start with **Summary Mode** to identify major bottlenecks.
- Use **Metrics Mode** to analyze specific inefficiencies (e.g., warp divergence, memory throughput).
- Keep your kernels optimized before profiling to avoid unnecessary overhead.
- Use **Nsight Compute** (newer tool) for advanced profiling beyond `nvprof`.

-----

# Parallel Resuction as **SYNCHRONIZATION** Example
- `cudaDeviceSynchronize`
  - provides global synchronization between host and device
  - in cuda, often asychronous calls like kernel launches are made from the host
- `__syncthreads`
  - provides the synchronization within a block
  - **this function should call from only device code**, and it will force threads within a block to wait until all the threads in that particular block to reach that particular point in code
  -  global and shared memory access made by all the threads in the block prior to the synchronization point or syncthread() function call will be visible to other threads in the block after the synchronization point
  - remember this synchronization happens between threads within a block

-----

# **Usage of `__syncthreads()` in Parallel Reduction Algorithm**
The general problem of performing a **commutative** and **associative operation** (like summation, multiplication, etc.) across a vector is referred to as the **reduction problem**. This operation is fundamental in parallel computing, especially for aggregating data efficiently.

---

## **Sequential Reduction: Example**
In a sequential reduction, the operation is performed iteratively over the entire array:

```cpp
int sum = 0;

for (int i = 0; i < size; i++) {
    sum += array[i];
}
```
While straightforward, this approach processes elements one at a time and is computationally expensive for large datasets.

---

## **Parallel Reduction Approach**
Parallel reduction divides the workload among multiple threads, making it significantly faster by leveraging the parallelism offered by GPUs.

### **Steps in Parallel Reduction**
1. **Partitioning**:
   - The input vector is divided into smaller chunks (e.g., blocks of threads in CUDA).
2. **Chunk Reduction**:
   - Each chunk is processed independently by threads within a block, calculating partial sums.
3. **Final Reduction**:
   - The partial results from all blocks are aggregated to compute the final result.

### **Key Advantage**:
By parallelizing the operation, the time complexity is reduced from **O(n)** (sequential) to approximately **O(log(n))** (parallel).

---

## **Reduction Technique: Neighbored Pair Approach**
This technique reduces the vector iteratively, with threads working on neighboring elements based on an **offset** value.

### **Algorithm Description**
- **Step 1**: Start with an **offset** of 1.
  - Each thread processes a pair of neighboring elements (e.g., `tid` and `tid + offset`).
- **Step 2**: Double the **offset** in each iteration.
  - Threads that are multiples of `2 * offset` perform the computation.
- **Step 3**: Repeat until the offset exceeds the block size.
  - This process ensures that the vector is reduced progressively to a single result within each block.

### **Thread Synchronization with `__syncthreads()`**
In each iteration:
- Threads must synchronize using `__syncthreads()` to ensure that all threads complete their current computation before moving to the next offset.
- This avoids race conditions where some threads might read updated values while others are still working.

---

### **Code Implementation**
Here’s a CUDA code snippet illustrating the neighbored pair reduction approach:

```cpp
for (int offset = 1; offset < blockDim.x; offset *= 2) {
    // Only threads with IDs that are multiples of 2 * offset perform the addition
    if (tid % (2 * offset) == 0) {
        input[tid] += input[tid + offset]; // Add the value of the neighboring element
    }

    // Synchronize all threads in the block before proceeding to the next iteration
    __syncthreads();
}
```

### **Explanation of Key Points**
1. **Offset Logic**:
   - Initially, threads with IDs `0, 2, 4, 6, ...` (multiples of `2 * offset`) process pairs.
   - As the offset increases (e.g., `1 → 2 → 4 → 8`), fewer threads perform computation, but each processes larger strides of the array.

2. **Thread Synchronization**:
   - `__syncthreads()` ensures that all threads complete their work for a given offset before moving to the next.
   - This prevents data inconsistency and ensures correctness of the reduction.

3. **Thread Limitation**:
   - Each iteration dynamically limits the threads performing summation, based on the current offset value.

---

### **Benefits of Using `__syncthreads()` in Reduction**
- **Data Integrity**: Ensures all threads in a block are synchronized before proceeding.
- **Parallel Efficiency**: Allows intermediate results to be reused in subsequent iterations without errors.
- **Scalability**: Enables efficient reduction for large datasets by processing data in logarithmic steps.

---

### **Visualization of Neighbored Pair Reduction**
Let’s consider an example with `8` elements and `blockDim.x = 8`:

| Iteration | Offset | Active Threads | Computation                                  |
|-----------|--------|----------------|---------------------------------------------|
| 1         | 1      | 0, 2, 4, 6     | `input[0] += input[1], input[2] += input[3]` |
| 2         | 2      | 0, 4           | `input[0] += input[2], input[4] += input[6]` |
| 3         | 4      | 0              | `input[0] += input[4]`                       |

At the end of the process, `input[0]` contains the sum of all elements.

---

### **Conclusion**
The use of `__syncthreads()` is crucial in parallel reduction algorithms to ensure thread safety and correct results. The neighbored pair approach, though simple, is foundational for understanding more advanced reduction techniques like warp-based or shared-memory reductions. This example highlights the importance of synchronization and efficient thread utilization in CUDA programming.

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Function prototype for CPU-based reduction
int reduction_cpu(int* input, const int size);

// Function prototype for comparing CPU and GPU results
void compare_results(int gpu_result, int cpu_result);

// CUDA kernel prototype for reduction on GPU
__global__ void reduction_neighbored_pairs(int* input, int* temp, int size);

// Helper function prototype to initialize an array
void initialize(int* array, int size, int init_type);

#define INIT_RANDOM 1 // Constant for random initialization type

int main(int argc, char** argv) {
    printf("Running neighbored pairs reduction kernel\n");

    int size = 1 << 27; // Number of elements in the array (128 MB of data)
    int byte_size = size * sizeof(int); // Total memory size in bytes
    int block_size = 128; // Number of threads per block

    int* h_input; // Host input array
    int* h_ref;   // Host temporary array for GPU results

    // Allocate memory for the input array on the host
    h_input = (int*)malloc(byte_size);

    // Initialize the input array with random values
    initialize(h_input, size, INIT_RANDOM);

    // Perform reduction on the CPU and get the result
    int cpu_result = reduction_cpu(h_input, size);

    // Define CUDA grid and block dimensions
    dim3 block(block_size);                         // Block dimension
    dim3 grid((size + block.x - 1) / block.x);      // Grid dimension, ensuring full coverage of the array

    printf("Kernel launch parameters | grid.x: %d, block.x: %d\n", grid.x, block.x);

    int temp_array_byte_size = sizeof(int) * grid.x; // Size for temporary array to store block results

    // Allocate memory for the temporary result array on the host
    h_ref = (int*)malloc(temp_array_byte_size);

    int* d_input; // Device input array pointer
    int* d_temp;  // Device temporary array pointer

    // Allocate device memory for the input array
    cudaMalloc((void**)&d_input, byte_size);

    // Allocate device memory for the temporary result array
    cudaMalloc((void**)&d_temp, temp_array_byte_size);

    // Initialize the temporary device array to zeros
    cudaMemset(d_temp, 0, temp_array_byte_size);

    // Copy the input data from the host to the device
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    // Launch the GPU reduction kernel
    reduction_neighbored_pairs<<<grid, block>>>(d_input, d_temp, size);

    // Wait for GPU computations to complete
    cudaDeviceSynchronize();

    // Copy the results from the device back to the host
    cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost);

    // Perform the final reduction on the host
    int gpu_result = 0;
    for (int i = 0; i < grid.x; i++) {
        gpu_result += h_ref[i]; // Sum all block results to get the final result
    }

    // Compare the GPU result with the CPU result
    compare_results(gpu_result, cpu_result);

    // Free host memory
    free(h_input);
    free(h_ref);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_temp);

    // Reset the CUDA device
    cudaDeviceReset();

    return 0;
}

// CPU-based reduction implementation
int reduction_cpu(int* input, const int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += input[i]; // Accumulate all array elements
    }
    return sum; // Return the final sum
}

// Compare results from GPU and CPU
void compare_results(int gpu_result, int cpu_result) {
    printf("GPU Result: %d, CPU Result: %d\n", gpu_result, cpu_result);
    if (gpu_result == cpu_result) {
        printf("Results match!\n"); // Print success if results are the same
    } else {
        printf("Results do NOT match!\n"); // Print error if results differ
    }
}

// CUDA kernel for reduction using neighbored pairs
__global__ void reduction_neighbored_pairs(int* input, int* temp, int size) {
    int tid = threadIdx.x;                           // Thread ID within the block
    int gid = blockDim.x * blockIdx.x + threadIdx.x; // Global ID across all blocks

    // Check if the thread is within bounds
    if (gid >= size) {
        return;
    }

    // Perform reduction within a block using neighboring pairs
    for (int offset = 1; offset <= blockDim.x / 2; offset *= 2) {
        if (tid % (2 * offset) == 0 && gid + offset < size) {
            input[gid] += input[gid + offset]; // Add neighboring element
        }
        __syncthreads(); // Synchronize threads within the block
    }

    // Store the block's result in the temp array
    if (tid == 0) {
        temp[blockIdx.x] = input[gid]; // First thread in the block writes the result
    }
}

// Initialize the input array
void initialize(int* array, int size, int init_type) {
    if (init_type == INIT_RANDOM) {
        for (int i = 0; i < size; i++) {
            array[i] = rand() % 100; // Fill the array with random values between 0 and 99
        }
    }
}
```

---

# Condition Comparison: `offset <= blockDim.x /2` VS `offset < blockDim.x`

The difference between the conditions `offset <= blockDim.x / 2` and `offset < blockDim.x` lies in **how far the reduction process continues** and **when threads are no longer active**. Let’s analyze both conditions in detail, focusing on their implications for the reduction process.

---

### **Condition: `offset <= blockDim.x / 2`**

#### **How It Works:**
- **Stopping Point**: This condition ensures that the offset never exceeds half the block size (`blockDim.x / 2`).
- **Why Half?**
  - In the reduction algorithm, the neighbors being summed are at an increasing distance (`offset`) from each other.
  - When the offset becomes greater than `blockDim.x / 2`, some threads (e.g., those in the second half of the block) no longer have a valid neighbor (`tid + offset` would go out of bounds).

#### **Example (`blockDim.x = 8`):**
| **Offset** | **Active Threads**  | **Thread Pairing (Summing)** |
|------------|----------------------|------------------------------|
| 1          | All threads (0–7)    | `0+1, 2+3, 4+5, 6+7`        |
| 2          | Threads 0, 2, 4, 6   | `0+2, 4+6`                  |
| 4          | Threads 0, 4         | `0+4`                       |

- When `offset = 4`, only thread `tid = 0` performs work on the pair `0+4`.
- **Stops when `offset = 8`**, as this exceeds `blockDim.x / 2`.

#### **Implications:**
- **Efficient Thread Usage**: Ensures that threads are active only when their neighbors exist within the block.
- **Prevents Redundant Computations**: Threads beyond `blockDim.x / 2` would not contribute to the reduction because they would no longer have valid neighbors.

---

### **Condition: `offset < blockDim.x`**

#### **How It Works:**
- **Stopping Point**: This condition allows the offset to go all the way up to `blockDim.x - 1`.
- **What Happens Beyond `blockDim.x / 2`?**
  - For offsets larger than `blockDim.x / 2`, only thread `tid = 0` continues working (since all other threads are effectively out of range).
  - Thread `tid = 0` would repeatedly sum values that might already have been reduced earlier.

#### **Example (`blockDim.x = 8`):**
| **Offset** | **Active Threads**  | **Thread Pairing (Summing)** |
|------------|----------------------|------------------------------|
| 1          | All threads (0–7)    | `0+1, 2+3, 4+5, 6+7`        |
| 2          | Threads 0, 2, 4, 6   | `0+2, 4+6`                  |
| 4          | Threads 0, 4         | `0+4`                       |
| 8          | Only thread 0        | `0` (No valid neighbor)     |

#### **Implications:**
- **Unnecessary Iterations**: Beyond `offset = blockDim.x / 2`, all threads except thread `tid = 0` stop contributing, making iterations inefficient.
- **Redundancy**: Thread `tid = 0` will redundantly attempt to sum elements that are already part of the result.

---

### **Comparison: `offset <= blockDim.x / 2` vs `offset < blockDim.x`**

| Aspect                | `offset <= blockDim.x / 2`                        | `offset < blockDim.x`                              |
|-----------------------|---------------------------------------------------|---------------------------------------------------|
| **Stopping Condition** | Stops when the offset exceeds half the block size.| Continues until the offset is equal to block size.|
| **Thread Activity**    | Halves active threads in each iteration efficiently.| Threads beyond `blockDim.x / 2` do no meaningful work.|
| **Performance**        | More efficient, as it avoids redundant iterations.| Less efficient due to unnecessary iterations.      |
| **Safety**             | Always safe and avoids out-of-bounds memory access.| Potential for redundant or incorrect access checks.|

---

### **Which Condition Should You Use?**

- **`offset <= blockDim.x / 2`** is preferred because:
  1. It ensures that only valid threads perform computations.
  2. It avoids unnecessary iterations and redundant calculations.
  3. It aligns with the reduction process's structure, where the number of active threads is halved in each step.

- **`offset < blockDim.x`** might only be appropriate in cases where you need to consider operations across the entire block, even after most threads are inactive, which is rare in reduction algorithms.

---

### **Summary**

Use **`offset <= blockDim.x / 2`** for efficient and logically correct parallel reduction. This condition ensures:
1. No out-of-bounds memory access.
2. Optimal use of threads.
3. Reduced computational overhead.

-----

# Divergence in Reduction Algorithm

In our previous parallel reduction algorithm, unfortunately, we have warp divergence.

For example, in the first iteration only the threads with even global indices perform work, and in the second, only threads that have gobal indices that are multiples of 4 perform work. Only in the last several iterations, the the warp divergence is insignificant or non-existant.

-----

# Solution 1: Forcing neighboring threads to perform summation

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define INIT_RANDOM 1

// Function prototypes
void initialize(int* array, int size, int init_type);
int reduction_cpu(int* input, const int size);

__global__ void reduction_neighbored_pairs_improved(int* int_array, int* temp_array, int size) {
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    // Local data block pointer
    int* i_data = int_array + blockDim.x * blockIdx.x;

    if (gid >= size) {
        return;
    }

    // Perform reduction using neighbored pairs
    for (int offset = 1; offset <= blockDim.x / 2; offset *= 2) {
        int index = 2 * offset * tid;

        if (index < blockDim.x) {
            i_data[index] += i_data[index + offset];
        }

        __syncthreads();
    }

    // Store the result of this block in the temp array
    if (tid == 0) {
        temp_array[blockIdx.x] = i_data[0];
    }
}

int main(int argc, char** argv) {
    printf("Running improved neighbored pairs reduction kernel\n");

    int size = 1 << 27; // Number of elements in the array (128 MB of data)
    int byte_size = size * sizeof(int);
    int block_size = 128; // Number of threads per block

    int* h_input; // Host input array
    int* h_ref;   // Host temporary array for GPU results

    // Allocate memory for the input array on the host
    h_input = (int*)malloc(byte_size);

    // Initialize the input array with random values
    initialize(h_input, size, INIT_RANDOM);

    // Perform reduction on the CPU and get the result
    int cpu_result = reduction_cpu(h_input, size);
    printf("CPU Result: %d\n", cpu_result);

    // Define CUDA grid and block dimensions
    dim3 block(block_size);
    dim3 grid((size + block.x - 1) / block.x);

    printf("Kernel launch parameters | grid.x: %d, block.x: %d\n", grid.x, block.x);

    int temp_array_byte_size = sizeof(int) * grid.x; // Size for temporary array to store block results

    // Allocate memory for the temporary result array on the host
    h_ref = (int*)malloc(temp_array_byte_size);

    int* d_input; // Device input array pointer
    int* d_temp;  // Device temporary array pointer

    // Allocate device memory for the input array
    cudaMalloc((void**)&d_input, byte_size);

    // Allocate device memory for the temporary result array
    cudaMalloc((void**)&d_temp, temp_array_byte_size);

    // Initialize the temporary device array to zeros
    cudaMemset(d_temp, 0, temp_array_byte_size);

    // Copy the input data from the host to the device
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    // Launch the GPU reduction kernel
    reduction_neighbored_pairs_improved<<<grid, block>>>(d_input, d_temp, size);

    // Wait for GPU computations to complete
    cudaDeviceSynchronize();

    // Copy the results from the device back to the host
    cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost);

    // Perform the final reduction on the host
    int gpu_result = 0;
    for (int i = 0; i < grid.x; i++) {
        gpu_result += h_ref[i]; // Sum all block results to get the final result
    }

    printf("GPU Result: %d\n", gpu_result);

    // Free host memory
    free(h_input);
    free(h_ref);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_temp);

    // Reset the CUDA device
    cudaDeviceReset();

    return 0;
}

// CPU-based reduction implementation
int reduction_cpu(int* input, const int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += input[i]; // Accumulate all array elements
    }
    return sum; // Return the final sum
}

// Initialize the input array
void initialize(int* array, int size, int init_type) {
    if (init_type == INIT_RANDOM) {
        for (int i = 0; i < size; i++) {
            array[i] = rand() % 100; // Fill the array with random values between 0 and 99
        }
    }
}
```

-----

# Study Notes: Improved Neighbored Pairs Reduction Algorithm

This document explains the improvements made in the updated parallel reduction algorithm using CUDA. The improvements focus on optimizing memory access patterns and computational efficiency.

## Overview of the Algorithm

The improved algorithm performs parallel reduction by:

1. Using shared memory more effectively to reduce global memory access.
2. Reducing the number of active threads iteratively, which minimizes unnecessary calculations.
3. Employing synchronization to ensure correct intermediate results within a block.

### Key Features

1. **Block-level Pointer Optimization**:
   ```cpp
   int* i_data = int_array + blockDim.x * blockIdx.x;
   ```
   - This creates a local pointer to the block's data in the input array.
   - Reduces redundant global memory access, as threads only work with data relevant to their block.

2. **Optimized Reduction Loop**:
   ```cpp
   for (int offset = 1; offset <= blockDim.x / 2; offset *= 2) {
       int index = 2 * offset * tid;

       if (index < blockDim.x) {
           i_data[index] += i_data[index + offset];
       }
       __syncthreads();
   }
   ```
   - The loop ensures that only active threads perform the required addition.
   - The number of active threads is halved at each iteration, reducing computation overhead.
   - Synchronization (`__syncthreads()`) ensures that all threads complete their computation before the next iteration starts.

3. **Efficient Storage of Partial Results**:
   ```cpp
   if (tid == 0) {
       temp_array[blockIdx.x] = i_data[0];
   }
   ```
   - Only one thread (the first thread in the block) writes the result back to the temporary array.
   - This minimizes memory usage and avoids race conditions.

## Improvements Compared to the Old Algorithm

| Aspect                | Old Algorithm                              | Improved Algorithm                        |
|-----------------------|--------------------------------------------|-------------------------------------------|
| **Memory Access**     | Accessed global memory for all operations | Uses block-local pointers to reduce access |
| **Thread Efficiency** | All threads performed computations         | Halves active threads at each iteration   |
| **Synchronization**   | May lead to redundant synchronization      | Synchronization only when needed          |
| **Result Storage**    | Global memory writes for all threads       | Single thread stores the block result     |

### Performance Gains

- **Reduced Global Memory Access**: By working with a block-local pointer, threads avoid unnecessary global memory reads and writes, which is the primary bottleneck in CUDA.
- **Minimized Active Threads**: Reducing the number of active threads at each step ensures only the necessary computation is performed, leading to better utilization of GPU resources.
- **Synchronization Efficiency**: The synchronization ensures correctness without introducing overhead from redundant `__syncthreads()` calls.

## Visual Representation of the Algorithm

1. **Initial State**:
   - All threads work on the data.
   - Each thread is assigned a unique portion of the input array.

2. **Reduction Steps**:
   - Threads combine neighboring elements iteratively.
   - The number of active threads is halved after each step.

3. **Final State**:
   - The first thread in each block stores the final reduced value in a temporary array.

## Code Snippet Highlight

### Reduction Kernel
```cpp
__global__ void reduction_neighbored_pairs_improved(int* int_array, int* temp_array, int size) {
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    // Local data block pointer
    int* i_data = int_array + blockDim.x * blockIdx.x;

    if (gid >= size) {
        return;
    }

    // Perform reduction using neighbored pairs
    for (int offset = 1; offset <= blockDim.x / 2; offset *= 2) {
        int index = 2 * offset * tid;

        if (index < blockDim.x) {
            i_data[index] += i_data[index + offset];
        }

        __syncthreads();
    }

    // Store the result of this block in the temp array
    if (tid == 0) {
        temp_array[blockIdx.x] = i_data[0];
    }
}
```

## Conclusion

The improved algorithm leverages better memory access patterns, reduces unnecessary computations, and ensures efficient synchronization. These enhancements result in better performance and scalability on modern GPUs.

-----

# Setup of `i_data`

The `i_data` pointer is a local pointer that simplifies and optimizes access to a specific segment of the global input array. Here's a detailed explanation of why it is set up this way:

### Code Context
```cpp
int* i_data = int_array + blockDim.x * blockIdx.x;
```

### Purpose of `i_data`

1. **Block-specific Memory Access**:
   - CUDA organizes threads into blocks. Each block processes a specific portion of the input array.
   - `blockDim.x * blockIdx.x` calculates the starting index in the global `int_array` for the current block.
   - The `i_data` pointer points to the beginning of the current block's portion of the input array.

2. **Simplifies Indexing**:
   - Within the kernel, threads in a block only work with their local portion of the input array.
   - By setting `i_data` to the starting address of the block’s data, we can use thread indices (`tid`) for direct indexing.
   - Instead of calculating the global index repeatedly, we access data relative to the block’s start:
     ```cpp
     i_data[tid] // accesses the element assigned to the current thread
     ```
   - This is simpler and avoids recalculating offsets repeatedly during the reduction loop.

3. **Improves Memory Access Patterns**:
   - Threads in CUDA access memory in **warps** (groups of 32 threads). For optimal performance, threads should access memory in a contiguous manner (coalesced access).
   - Using `i_data` ensures that all threads in a block access contiguous memory addresses within the block’s portion, which leads to better memory coalescing and reduced latency.

4. **Minimizes Global Memory Access Overhead**:
   - Accessing global memory is slower compared to shared or local memory.
   - By creating a local pointer `i_data`, the kernel can reuse data within the block more efficiently, particularly when combined with synchronization and iterative reduction steps.

### Why Not Use the Global `int_array` Directly?

If we accessed the global `int_array` directly, every access would require calculating the global index:
```cpp
int global_index = blockIdx.x * blockDim.x + threadIdx.x;
int_array[global_index] += int_array[global_index + offset];
```

This approach:
- Redundantly calculates `blockIdx.x * blockDim.x` for every thread.
- Makes the code harder to read and maintain.
- Potentially increases memory access overhead.

By introducing `i_data`, these problems are avoided.

### Summary of Benefits
- **Simplifies indexing**: Thread-local access becomes straightforward with `i_data[tid]`.
- **Improves coalesced memory access**: Threads access contiguous memory locations within a block.
- **Reduces redundant calculations**: Avoids recalculating block offsets repeatedly.
- **Optimizes memory hierarchy**: Ensures faster access patterns within the GPU’s memory architecture.

### Visualization
Suppose `int_array` has 16 elements, `blockDim.x = 4`, and `gridDim.x = 4` (4 blocks, 4 threads each):
- Block 0: `int_array[0]` to `int_array[3]`
- Block 1: `int_array[4]` to `int_array[7]`
- Block 2: `int_array[8]` to `int_array[11]`
- Block 3: `int_array[12]` to `int_array[15]`

In each block:
- `i_data` points to the start of the respective range (e.g., `i_data = &int_array[4]` for Block 1).
- Threads access data as `i_data[0]`, `i_data[1]`, etc., instead of recalculating global indices repeatedly.

-----

# Solutionn 2: Interleaved Pairs

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void reduction_interleaved_pairs(int* int_array, int* temp_array, int size) {
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid >= size) {
        return;
    }

    // Perform reduction using interleaved pairs
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset && gid + offset < size) {
            int_array[gid] += int_array[gid + offset];
        }
        __syncthreads();
    }

    // Store the result of this block in the temp array
    if (tid == 0) {
        temp_array[blockIdx.x] = int_array[gid];
    }
}

int main(int argc, char** argv) {
    printf("Running interleaved pairs reduction kernel\n");

    int size = 1 << 27; // Number of elements in the array (128 MB of data)
    int byte_size = size * sizeof(int);
    int block_size = 128; // Number of threads per block

    int* h_input; // Host input array
    int* h_ref;   // Host temporary array for GPU results

    // Allocate memory for the input array on the host
    h_input = (int*)malloc(byte_size);

    // Initialize the input array with random values
    for (int i = 0; i < size; i++) {
        h_input[i] = rand() % 100;
    }

    // Define CUDA grid and block dimensions
    dim3 block(block_size);
    dim3 grid((size + block.x - 1) / block.x);

    printf("Kernel launch parameters | grid.x: %d, block.x: %d\n", grid.x, block.x);

    int temp_array_byte_size = sizeof(int) * grid.x; // Size for temporary array to store block results

    // Allocate memory for the temporary result array on the host
    h_ref = (int*)malloc(temp_array_byte_size);

    int* d_input; // Device input array pointer
    int* d_temp;  // Device temporary array pointer

    // Allocate device memory for the input array
    cudaMalloc((void**)&d_input, byte_size);

    // Allocate device memory for the temporary result array
    cudaMalloc((void**)&d_temp, temp_array_byte_size);

    // Initialize the temporary device array to zeros
    cudaMemset(d_temp, 0, temp_array_byte_size);

    // Copy the input data from the host to the device
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    // Launch the GPU reduction kernel
    reduction_interleaved_pairs<<<grid, block>>>(d_input, d_temp, size);

    // Wait for GPU computations to complete
    cudaDeviceSynchronize();

    // Copy the results from the device back to the host
    cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost);

    // Perform the final reduction on the host
    int gpu_result = 0;
    for (int i = 0; i < grid.x; i++) {
        gpu_result += h_ref[i]; // Sum all block results to get the final result
    }

    // Perform reduction on the CPU for verification
    int cpu_result = 0;
    for (int i = 0; i < size; i++) {
        cpu_result += h_input[i];
    }

    // Compare results
    printf("GPU Result: %d, CPU Result: %d\n", gpu_result, cpu_result);
    if (gpu_result == cpu_result) {
        printf("Results match!\n");
    } else {
        printf("Results do NOT match!\n");
    }

    // Free host memory
    free(h_input);
    free(h_ref);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_temp);

    // Reset the CUDA device
    cudaDeviceReset();

    return 0;
}
```

-----

# Study Notes: Interleaved Pairs Reduction Algorithm

This document explains the interleaved pairs reduction algorithm implemented using CUDA. The algorithm is optimized for parallel reduction on GPU, efficiently summing large arrays by leveraging interleaved data access patterns and thread synchronization.

## Overview of the Algorithm

The interleaved pairs reduction algorithm performs parallel reduction by:

1. Dividing the input data into blocks, with each block processed by a set of threads.
2. Halving the active threads at each iteration, reducing redundant computation.
3. Synchronizing threads within a block to ensure correctness during intermediate calculations.
4. Storing partial sums for each block, which are later combined on the host.

### Key Features

1. **Iterative Reduction**:
   ```cpp
   for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
       if (tid < offset && gid + offset < size) {
           int_array[gid] += int_array[gid + offset];
       }
       __syncthreads();
   }
   ```
   - The number of active threads decreases by half in each iteration.
   - Threads add elements separated by `offset` and store the result in the current position.
   - Synchronization ensures that all threads finish their updates before the next iteration.

2. **Efficient Memory Access**:
   - Threads access data contiguously, improving memory coalescing and reducing global memory access overhead.
   - The algorithm uses shared memory for temporary storage, although in this version, it operates directly on the global memory array.

3. **Final Block Result Storage**:
   ```cpp
   if (tid == 0) {
       temp_array[blockIdx.x] = int_array[gid];
   }
   ```
   - The first thread in each block writes the block's final result to a temporary array.
   - This minimizes memory usage and avoids race conditions.

## Improvements Compared to Naive Algorithms

| Aspect                | Naive Algorithm                          | Interleaved Pairs Algorithm                |
|-----------------------|------------------------------------------|--------------------------------------------|
| **Memory Access**     | Inefficient, redundant global access     | Optimized, coalesced memory access         |
| **Thread Utilization**| All threads active throughout            | Threads reduced by half each iteration    |
| **Synchronization**   | Redundant or absent synchronization      | Synchronization ensures correctness        |
| **Result Storage**    | All threads write results individually   | Only one thread stores the block result    |

## Kernel Function Explanation

### Reduction Loop

The main loop reduces the array using interleaved pairs:

- `offset` starts at half the block size and halves in each iteration.
- Threads at indices `tid` less than `offset` perform an addition with their counterpart at `gid + offset`.
- After each iteration, `__syncthreads()` ensures all threads finish their updates before proceeding.

### Result Storage

After the reduction, only the first thread in each block writes the block’s result to the `temp_array`. This avoids conflicts and ensures minimal memory usage.

### Edge Case Handling

The algorithm checks boundary conditions to prevent out-of-bound memory access:

```cpp
if (gid >= size) {
    return;
}
```

## Host Code Workflow

1. **Memory Allocation**:
   - Host allocates memory for the input array and temporary result array.
   - Device memory is allocated for input and intermediate results.

2. **Kernel Launch**:
   - The grid and block dimensions are configured based on the array size and block size.
   - The reduction kernel processes the input array in parallel.

3. **Final Reduction**:
   - The host performs a final summation of block results to obtain the total sum.

## Code Example

### Kernel Function
```cpp
__global__ void reduction_interleaved_pairs(int* int_array, int* temp_array, int size) {
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid >= size) {
        return;
    }

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset && gid + offset < size) {
            int_array[gid] += int_array[gid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        temp_array[blockIdx.x] = int_array[gid];
    }
}
```

### Host Code Highlights

- Memory setup:
   ```cpp
   int* d_input;
   cudaMalloc((void**)&d_input, byte_size);
   cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);
   ```

- Kernel launch:
   ```cpp
   reduction_interleaved_pairs<<<grid, block>>>(d_input, d_temp, size);
   cudaDeviceSynchronize();
   ```

- Final reduction:
   ```cpp
   for (int i = 0; i < grid.x; i++) {
       gpu_result += h_ref[i];
   }
   ```

## Advantages

1. **Scalability**: Handles large arrays efficiently by splitting work across multiple blocks.
2. **Performance**: Reduces global memory access and computational overhead.
3. **Correctness**: Ensures accurate intermediate results with synchronization.

## Limitations

- **Shared Memory Usage**: The current implementation operates directly on global memory, which may not fully utilize shared memory for performance.
- **Warp Divergence**: Threads within a warp may execute divergent code paths due to boundary conditions.

## Conclusion

The interleaved pairs reduction algorithm optimizes parallel summation by leveraging efficient memory access, synchronization, and reduced computation. It is a foundational approach in CUDA programming and serves as a basis for more advanced reduction techniques.

-----

# Unrolling IN CUDA

Goal of Unrolling: Improving the performance by reducing instruction overhead and creating more independent instructions to schedule

-----

# Loop Unrolling
  - In loop unrolling, rather than writing the body of a loop once and using a loop to execute it repeatedly, the body is written in code multiple times.
  - The number of copies made of the loop bodies is called loop unrolling factor.

# Thread Block Unrolling

Unrolling we are performing is block thread unrolling, but the goal, reducing instruction overhead, remains the same.

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA Kernel for reduction using block unrolling
__global__ void reduction_unrolling_blocks2(int* input, int* temp, int size) {
    int tid = threadIdx.x; // Thread index within the block
    int BLOCK_OFFSET = blockIdx.x * blockDim.x * 2; // Offset for each block, considering unrolling by 2
    int index = BLOCK_OFFSET + tid; // Global index of the thread

    // Pointer to the start of the block's data
    int* i_data = input + BLOCK_OFFSET;

    // Unroll the first addition step by combining elements separated by blockDim.x
    if ((index + blockDim.x) < size) {
        input[index] += input[index + blockDim.x];
    }

    __syncthreads(); // Ensure all threads complete the first addition before proceeding

    // Perform standard reduction within the block
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            i_data[tid] += i_data[tid + offset];
        }

        __syncthreads(); // Synchronize threads after each reduction step
    }

    // Write the block's final result to the temporary array
    if (tid == 0) {
        temp[blockIdx.x] = i_data[0];
    }
}

int main(int argc, char** argv) {
    printf("Running reduction with block unrolling (factor of 2)\n");

    int size = 1 << 27; // Number of elements in the array (128 MB of data)
    int byte_size = size * sizeof(int);
    int block_size = 128; // Number of threads per block

    int* h_input; // Host input array
    int* h_temp;  // Host temporary array for GPU results

    // Allocate memory for the input array on the host
    h_input = (int*)malloc(byte_size);

    // Initialize the input array with random values
    for (int i = 0; i < size; i++) {
        h_input[i] = rand() % 100;
    }

    // Define CUDA grid and block dimensions
    dim3 block(block_size);
    dim3 grid((size + block.x * 2 - 1) / (block.x * 2)); // Grid dimensions considering unrolling

    printf("Kernel launch parameters | grid.x: %d, block.x: %d\n", grid.x, block.x);

    int temp_array_byte_size = sizeof(int) * grid.x; // Size for temporary array to store block results

    // Allocate memory for the temporary result array on the host
    h_temp = (int*)malloc(temp_array_byte_size);

    int* d_input; // Device input array pointer
    int* d_temp;  // Device temporary array pointer

    // Allocate device memory for the input array
    cudaMalloc((void**)&d_input, byte_size);

    // Allocate device memory for the temporary result array
    cudaMalloc((void**)&d_temp, temp_array_byte_size);

    // Initialize the temporary device array to zeros
    cudaMemset(d_temp, 0, temp_array_byte_size);

    // Copy the input data from the host to the device
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    // Launch the GPU reduction kernel
    reduction_unrolling_blocks2<<<grid, block>>>(d_input, d_temp, size);

    // Wait for GPU computations to complete
    cudaDeviceSynchronize();

    // Copy the results from the device back to the host
    cudaMemcpy(h_temp, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost);

    // Perform the final reduction on the host
    int gpu_result = 0;
    for (int i = 0; i < grid.x; i++) {
        gpu_result += h_temp[i]; // Sum all block results to get the final result
    }

    // Perform reduction on the CPU for verification
    int cpu_result = 0;
    for (int i = 0; i < size; i++) {
        cpu_result += h_input[i];
    }

    // Compare results
    printf("GPU Result: %d, CPU Result: %d\n", gpu_result, cpu_result);
    if (gpu_result == cpu_result) {
        printf("Results match!\n");
    } else {
        printf("Results do NOT match!\n");
    }

    // Free host memory
    free(h_input);
    free(h_temp);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_temp);

    // Reset the CUDA device
    cudaDeviceReset();

    return 0;
}
```

-----

# Study Notes: Reduction with Block Unrolling (Factor of 2)

This document explains the concept of block unrolling and its application in the `reduction_unrolling_blocks2` CUDA kernel.

## What is Unrolling?
Unrolling is an optimization technique used in parallel programming to improve performance by reducing the number of iterations in a loop. By performing multiple operations in a single iteration, we:

1. **Reduce Overhead**: Fewer loop control operations (e.g., increment, comparison).
2. **Increase Throughput**: More data is processed in a single iteration.
3. **Optimize Memory Access**: Combine adjacent memory accesses to improve coalescing.

In CUDA, unrolling is particularly useful in reduction algorithms, where adjacent elements are combined to compute partial sums. By unrolling, we can reduce the number of iterations required for the first step of reduction, thereby improving performance.

---

## Unrolling in the `reduction_unrolling_blocks2` Kernel

### Unrolling by a Factor of 2

The kernel performs block-level unrolling by a factor of 2. This means that each thread processes two elements of the input array in a single step. Here’s how it is applied:

1. **Calculate the Block Offset**:
   ```cpp
   int BLOCK_OFFSET = blockIdx.x * blockDim.x * 2;
   ```
   - Each block processes twice the number of threads it contains.
   - The block offset accounts for this by multiplying the block index (`blockIdx.x`) by `blockDim.x * 2`.

2. **Perform the First Addition Step**:
   ```cpp
   if ((index + blockDim.x) < size) {
       input[index] += input[index + blockDim.x];
   }
   ```
   - Each thread adds the element at its corresponding index (`index`) to the element that is `blockDim.x` positions ahead.
   - This reduces the array size for subsequent steps by a factor of 2, effectively unrolling the first iteration of the reduction loop.

3. **Synchronize Threads**:
   ```cpp
   __syncthreads();
   ```
   - Synchronization ensures that all threads complete their addition before moving on to the iterative reduction.

### Iterative Reduction
After unrolling the first step, the kernel performs a standard iterative reduction within the block:

```cpp
for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
        i_data[tid] += i_data[tid + offset];
    }
    __syncthreads();
}
```
- The number of active threads is halved in each iteration.
- Threads combine adjacent elements in the shared memory or local memory array.

### Final Block Result Storage
The first thread in the block stores the final reduced result for the block:

```cpp
if (tid == 0) {
    temp[blockIdx.x] = i_data[0];
}
```
This result is later combined by the host or another kernel to produce the final sum.

---

## Advantages of Unrolling

1. **Reduced Loop Iterations**:
   - By unrolling the first step, we halve the number of iterations required for the reduction.

2. **Improved Memory Coalescing**:
   - Threads access adjacent memory locations in the unrolling step, which improves memory bandwidth utilization.

3. **Higher Throughput**:
   - Fewer iterations and better memory access patterns result in higher overall throughput.

---

## Comparison: Without vs. With Unrolling

| Aspect                 | Without Unrolling                         | With Unrolling (Factor of 2)              |
|------------------------|-------------------------------------------|-------------------------------------------|
| **Iterations**         | One addition per thread per iteration     | Two additions per thread in the first step |
| **Memory Access**      | Single memory access per thread           | Two memory accesses per thread (contiguous) |
| **Performance**        | Higher overhead due to more iterations    | Reduced overhead, better memory utilization |

---

## Visualization

### Initial State
```
Input Array:
[ A, B, C, D, E, F, G, H ]
Threads:       T0  T1  T2  T3
```
- Threads T0 to T3 each work on two elements (unrolling).

### Unrolling Step (First Addition)
```
T0: A += E  ->  A+E
T1: B += F  ->  B+F
T2: C += G  ->  C+G
T3: D += H  ->  D+H
```
- After this step, the array is effectively reduced by half:
```
[ A+E, B+F, C+G, D+H ]
Threads:       T0   T1   T2   T3
```

### Iterative Reduction
1. **Step 1**:
```
T0: (A+E) += (C+G)  ->  (A+E) + (C+G)
T1: (B+F) += (D+H)  ->  (B+F) + (D+H)
```
- Result:
```
[ (A+E) + (C+G), (B+F) + (D+H) ]
Threads:       T0               T1
```

2. **Step 2**:
```
T0: [(A+E) + (C+G)] += [(B+F) + (D+H)]
```
- Final Result:
```
[ Final Sum ]
Thread:       T0
```

---

## Code Highlight

### Unrolling Step
```cpp
if ((index + blockDim.x) < size) {
    input[index] += input[index + blockDim.x];
}
```
- Each thread combines its element with the corresponding element that is `blockDim.x` positions ahead.
- This reduces the data size for subsequent reduction steps by a factor of 2.

### Offset-Based Reduction
```cpp
for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
        i_data[tid] += i_data[tid + offset];
    }
    __syncthreads();
}
```
- Standard reduction is performed after the unrolling step.
- The number of threads decreases by half in each iteration.

---

## Conclusion
The block unrolling technique in `reduction_unrolling_blocks2` improves the performance of the reduction kernel by processing more data in fewer iterations and optimizing memory access patterns. While unrolling by a factor of 2 is a simple and effective optimization, higher factors can be applied for further gains, depending on the problem size and available resources.

-----

# **Study Notes: Reduction with Block Unrolling (Factors of 2 and 4)**

This document explains the concept of block unrolling and its application in the `reduction_unrolling_blocks2` CUDA kernel. Additionally, we compare unrolling interleaved reduction with the regular interleaved reduction approach and discuss extending unrolling to a factor of 4.

## What is Unrolling?
Unrolling is an optimization technique used in parallel programming to improve performance by reducing the number of iterations in a loop. By performing multiple operations in a single iteration, we:

1. **Reduce Overhead**: Fewer loop control operations (e.g., increment, comparison).
2. **Increase Throughput**: More data is processed in a single iteration.
3. **Optimize Memory Access**: Combine adjacent memory accesses to improve coalescing.

In CUDA, unrolling is particularly useful in reduction algorithms, where adjacent elements are combined to compute partial sums. By unrolling, we can reduce the number of iterations required for the first step of reduction, thereby improving performance.

---

## Unrolling in the `reduction_unrolling_blocks2` Kernel

### Unrolling by a Factor of 2

The kernel performs block-level unrolling by a factor of 2. This means that each thread processes two elements of the input array in a single step. Here’s how it is applied:

1. **Calculate the Block Offset**:
   ```cpp
   int BLOCK_OFFSET = blockIdx.x * blockDim.x * 2;
   ```
   - Each block processes twice the number of threads it contains.
   - The block offset accounts for this by multiplying the block index (`blockIdx.x`) by `blockDim.x * 2`.

2. **Perform the First Addition Step**:
   ```cpp
   if ((index + blockDim.x) < size) {
       input[index] += input[index + blockDim.x];
   }
   ```
   - Each thread adds the element at its corresponding index (`index`) to the element that is `blockDim.x` positions ahead.
   - This reduces the array size for subsequent steps by a factor of 2, effectively unrolling the first iteration of the reduction loop.

3. **Synchronize Threads**:
   ```cpp
   __syncthreads();
   ```
   - Synchronization ensures that all threads complete their addition before moving on to the iterative reduction.

### Unrolling by a Factor of 4

When the unrolling factor is increased to 4, each thread processes four elements in the first step. Here’s how the kernel adapts:

1. **Calculate the Block Offset**:
   ```cpp
   int BLOCK_OFFSET = blockIdx.x * blockDim.x * 4;
   ```
   - Each block now processes four times the number of threads it contains.

2. **Perform the First Addition Step**:
   ```cpp
   if ((index + 3 * blockDim.x) < size) {
       int a1 = input[index];
       int a2 = input[index + blockDim.x];
       int a3 = input[index + 2 * blockDim.x];
       int a4 = input[index + 3 * blockDim.x];
       input[index] = a1 + a2 + a3 + a4;
   }
   ```
   - Each thread sums four elements, significantly reducing the array size in the first step.

3. **Synchronize Threads**:
   ```cpp
   __syncthreads();
   ```
   - Synchronization ensures correctness before moving to the iterative reduction.

### Iterative Reduction
The iterative reduction step remains the same, halving the number of active threads in each iteration:

```cpp
for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
        i_data[tid] += i_data[tid + offset];
    }
    __syncthreads();
}
```

---

## Comparison: Unrolling Interleaved (Factors of 2 and 4) vs. Regular Interleaved Reduction

| Aspect                   | Regular Interleaved Reduction           | Unrolling Interleaved (Factor of 2)       | Unrolling Interleaved (Factor of 4)       |
|--------------------------|------------------------------------------|------------------------------------------|------------------------------------------|
| **First Addition Step**  | Performed by all threads individually    | Threads process two elements initially    | Threads process four elements initially   |
| **Iterations**           | Full set of iterations based on data size| Reduced by one due to unrolling           | Reduced by two due to higher unrolling    |
| **Memory Access**        | Single memory access per thread          | Two memory accesses per thread (contiguous)| Four memory accesses per thread (contiguous)|
| **Performance**          | Higher due to increased iterations       | Improved due to reduced iterations         | Significantly improved for large arrays   |
| **Scalability**          | Good for small arrays                    | Better for larger arrays due to fewer iterations | Best for very large arrays                |

### Key Difference
- In **regular interleaved reduction**, each thread works on one element at a time, reducing the array size iteratively.
- In **unrolling interleaved reduction**, threads process multiple elements in the first step, reducing iterations and improving performance.

---

## Visualization

### Initial State (Factor of 4)
```
Input Array:
[ A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P ]
Threads:       T0  T1  T2  T3
```
- Threads T0 to T3 each work on four elements (unrolling).

### Unrolling Step (First Addition)
```
T0: A += E + I + M  ->  A+E+I+M
T1: B += F + J + N  ->  B+F+J+N
T2: C += G + K + O  ->  C+G+K+O
T3: D += H + L + P  ->  D+H+L+P
```
- After this step, the array is significantly reduced:
```
[ A+E+I+M, B+F+J+N, C+G+K+O, D+H+L+P ]
Threads:         T0         T1         T2         T3
```

### Iterative Reduction
1. **Step 1**:
```
T0: (A+E+I+M) += (C+G+K+O)  ->  (A+E+I+M) + (C+G+K+O)
T1: (B+F+J+N) += (D+H+L+P)  ->  (B+F+J+N) + (D+H+L+P)
```
- Result:
```
[ (A+E+I+M) + (C+G+K+O), (B+F+J+N) + (D+H+L+P) ]
Threads:         T0                            T1
```

2. **Step 2**:
```
T0: [(A+E+I+M) + (C+G+K+O)] += [(B+F+J+N) + (D+H+L+P)]
```
- Final Result:
```
[ Final Sum ]
Thread:         T0
```

---

## Code Highlight

### Unrolling Step for Factor of 4
```cpp
if ((index + 3 * blockDim.x) < size) {
    int a1 = input[index];
    int a2 = input[index + blockDim.x];
    int a3 = input[index + 2 * blockDim.x];
    int a4 = input[index + 3 * blockDim.x];
    input[index] = a1 + a2 + a3 + a4;
}
```
- Threads process four elements each, reducing the array size significantly in the first step.

### Offset-Based Reduction
```cpp
for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
        i_data[tid] += i_data[tid + offset];
    }
    __syncthreads();
}
```
- Standard reduction is performed after the unrolling step.
- The number of threads decreases by half in each iteration.

---

## Conclusion
The block unrolling technique in `reduction_unrolling_blocks2` improves the performance of the reduction kernel by processing more data in fewer iterations and optimizing memory access patterns. Unrolling by a factor of 4 provides even greater performance improvements, especially for very large arrays. This technique is highly scalable and can be extended further depending on the problem size and available resources.

-----

# Warp Unrolling

In the previous examples, we use
```cpp
for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
  if (tid < offset) {
    i_data[index] += i_data[index + offset];
  }

  __synthreads()
}
```
to perform accumulation of one thread block iteratively, and by implementing reduction using interleaved pair approach we got rif of most of the wrap divergence shown by the **neighbored pair approach**, but still for last five iterations in accumulation loop, first warp will exhibit warp divergence assuming the total number of elements is 128.

Scenario:
```
|--Warp 1--|--Warp 2--|--Warp 3--|--Warp 4--|

In the first iteration, only first two warps are active and the offset is 64, meaning first 64 threads will perform work.

In the second iteration, iteration is 32, therefore only first warp will perform accumulation.

For the iterations after, accumulation work will be performed inside the first warp.

This will cause divergence to occur in the first warp, and other warps are inactive.
```

However, we can avoid warp divergence in this scenario by using **Warp Unrolling**.

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void reduction_kernel_warp_unrolling(int* int_array, int* temp_array, int size) {
    int tid = threadIdx.x;

    // element index for this thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // local data pointer
    int* i_data = int_array + blockDim.x * blockIdx.x;

    for (int offset = blockDim.x / 2; offset >= 64; offset /= 2) {
        if (tid < offset) {
            i_data[tid] += i_data[tid + offset];
        }

        __syncthreads();
    }

    if (tid < 32) {
        // volatile specifier guarantees that memory load and store to the
        // global memory happen directly without using any cache
        volatile int* vsmem = i_data;

        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) {
        temp_array[blockIdx.x] = i_data[0];
    }
}

int main(int argc, char** argv) {
    printf("Running reduction kernel with warp unrolling\n");

    int size = 1 << 27; // Number of elements in the array (128 MB of data)
    int byte_size = size * sizeof(int);
    int block_size = 128; // Number of threads per block

    int* h_input; // Host input array
    int* h_temp;  // Host temporary array for GPU results

    // Allocate memory for the input array on the host
    h_input = (int*)malloc(byte_size);

    // Initialize the input array with random values
    for (int i = 0; i < size; i++) {
        h_input[i] = rand() % 100;
    }

    // Define CUDA grid and block dimensions
    dim3 block(block_size);
    dim3 grid((size + block.x - 1) / block.x);

    printf("Kernel launch parameters | grid.x: %d, block.x: %d\n", grid.x, block.x);

    int temp_array_byte_size = sizeof(int) * grid.x; // Size for temporary array to store block results

    // Allocate memory for the temporary result array on the host
    h_temp = (int*)malloc(temp_array_byte_size);

    int* d_input; // Device input array pointer
    int* d_temp;  // Device temporary array pointer

    // Allocate device memory for the input array
    cudaMalloc((void**)&d_input, byte_size);

    // Allocate device memory for the temporary result array
    cudaMalloc((void**)&d_temp, temp_array_byte_size);

    // Initialize the temporary device array to zeros
    cudaMemset(d_temp, 0, temp_array_byte_size);

    // Copy the input data from the host to the device
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    // Launch the GPU reduction kernel
    reduction_kernel_warp_unrolling<<<grid, block>>>(d_input, d_temp, size);

    // Wait for GPU computations to complete
    cudaDeviceSynchronize();

    // Copy the results from the device back to the host
    cudaMemcpy(h_temp, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost);

    // Perform the final reduction on the host
    int gpu_result = 0;
    for (int i = 0; i < grid.x; i++) {
        gpu_result += h_temp[i]; // Sum all block results to get the final result
    }

    // Perform reduction on the CPU for verification
    int cpu_result = 0;
    for (int i = 0; i < size; i++) {
        cpu_result += h_input[i];
    }

    // Compare results
    printf("GPU Result: %d, CPU Result: %d\n", gpu_result, cpu_result);
    if (gpu_result == cpu_result) {
        printf("Results match!\n");
    } else {
        printf("Results do NOT match!\n");
    }

    // Free host memory
    free(h_input);
    free(h_temp);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_temp);

    // Reset the CUDA device
    cudaDeviceReset();

    return 0;
}
```

-----

# Study Notes: Reduction with Warp Unrolling

This document provides a comprehensive explanation of the reduction algorithm with warp unrolling, including its key components, advantages, and execution flow visualization.

---

## What is Warp Unrolling?

Warp unrolling is an optimization technique used in CUDA to eliminate warp divergence during the reduction process by directly handling the last few iterations of the reduction loop for the first warp. This approach:

1. **Minimizes Warp Divergence**:
   - By avoiding conditional branching within a warp.
   - Ensures that all threads in the warp participate uniformly in the computations.

2. **Improves Performance**:
   - By reducing the overhead of synchronization and branching in the last few iterations.
   - Increases computational throughput.

---

## Algorithm Overview

### Key Steps:

1. **Block-Wide Reduction**:
   - Each block performs reduction on its portion of the input array iteratively.
   - Threads reduce elements in pairs, halving the active threads in each iteration.
   - Synchronization (`__syncthreads`) ensures correctness after each step.

2. **Handle the Last Warp**:
   - For offsets smaller than 64, the reduction is completed by a single warp.
   - The `volatile` keyword is used to access memory directly without caching to ensure immediate updates.

3. **Final Block Result Storage**:
   - The first thread in each block writes the block’s result to a temporary array.
   - The host combines these partial results to get the final sum.

---

### Key Code Elements

#### Iterative Reduction with Synchronization
```cpp
for (int offset = blockDim.x / 2; offset >= 64; offset /= 2) {
    if (tid < offset) {
        i_data[tid] += i_data[tid + offset];
    }
    __syncthreads();
}
```
- Reduces pairs of elements iteratively.
- Synchronization ensures correctness between iterations.

#### Warp-Level Reduction with `volatile`
```cpp
if (tid < 32) {
    volatile int* vsmem = i_data;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
}
```
- Directly computes the sum for the last warp.
- Ensures all threads in the warp participate without divergence.

#### Final Block-Wide Result Storage
```cpp
if (tid == 0) {
    temp_array[blockIdx.x] = i_data[0];
}
```
- Stores the final block result for aggregation on the host.

---

## What is `volatile`?

In CUDA, the `volatile` keyword ensures that memory operations are not cached and occur directly in global or shared memory. This is critical for:

1. **Ensuring Immediate Updates**:
   - Prevents threads from accessing stale data from registers or caches.

2. **Correctness in Warp-Level Reductions**:
   - Enables threads to see the most recent updates made by other threads.

### Example:
```cpp
volatile int* vsmem = i_data;
vsmem[tid] += vsmem[tid + 32];
```
- Forces each thread to directly read/write to shared memory without caching.
- Ensures consistency in the warp-wide reduction.

---

## Advantages of Warp Unrolling

1. **Reduced Warp Divergence**:
   - The last few iterations are handled uniformly within a single warp.

2. **Increased Performance**:
   - Fewer synchronization points and no branching.
   - Direct memory access via `volatile` improves execution efficiency.

3. **Scalability**:
   - Suitable for large arrays and high block sizes.

---

## Visualization of Execution Flow

### Test Case:
- Array Size: 128 elements.
- Block Size: 128 threads.

### Iterative Reduction (Offsets ≥ 64):

| Offset | Active Threads | Operation                              |
|--------|----------------|---------------------------------------|
| 64     | 64             | `T0 += T64, T1 += T65, ..., T63 += T127` |
| 32     | 32             | `T0 += T32, T1 += T33, ..., T31 += T63` |

### Warp-Level Reduction (Offsets < 64):

| Offset | Active Threads | Operation                              |
|--------|----------------|---------------------------------------|
| 16     | 32             | `T0 += T16, T1 += T17, ..., T15 += T31` |
| 8      | 32             | `T0 += T8, T1 += T9, ..., T7 += T15`   |
| 4      | 32             | `T0 += T4, T1 += T5, ..., T3 += T7`    |
| 2      | 32             | `T0 += T2, T1 += T3`                   |
| 1      | 32             | `T0 += T1`                             |

---

## Final Host Aggregation

After all blocks complete their reduction, the host combines the partial results stored in the `temp_array`:

```cpp
int gpu_result = 0;
for (int i = 0; i < grid.x; i++) {
    gpu_result += h_temp[i];
}
```

---

## Conclusion

Warp unrolling is an effective optimization for parallel reduction in CUDA. It eliminates warp divergence in the final iterations, improving overall efficiency and scalability. By combining block-wide reduction with warp-level unrolling, this technique ensures minimal overhead and high performance for large datasets.

-----

# Study Notes: Reduction with Complete Warp Unrolling

This document provides a comprehensive explanation of the reduction algorithm using **complete warp unrolling**, its implementation, benefits, and execution flow.

---

## What is Complete Warp Unrolling?

Complete warp unrolling is a technique that eliminates the iterative reduction loop by manually unrolling each step of the reduction process. This ensures:

1. **Minimized Loop Overhead**:
   - Reduces the cost of loop control (e.g., condition checks, increment operations).

2. **Improved Performance**:
   - Leverages CUDA's parallelism to process multiple reductions in fewer steps.

3. **Warp Efficiency**:
   - Handles the last warp (32 threads) directly without requiring synchronization or conditional branching.

---

## Key Steps in the Algorithm

### 1. Block-Level Reduction

- Each block processes a subset of the input array (`int_array`), reducing it to a single value.
- Threads work in pairs, halving the number of active threads at each step.

### 2. Unrolled Reduction Steps

- The reduction loop is replaced with explicitly written steps, each halving the active threads.
- Synchronization (`__syncthreads`) ensures correctness after each step.

### 3. Warp-Level Reduction

- The last warp (32 threads) performs the final reduction without requiring synchronization.
- The `volatile` keyword ensures consistent memory updates across threads.

### 4. Final Block Result Storage

- The first thread in each block writes the block’s result to the `temp_array` for host-side aggregation.

---

## Key Code Components

### Block-Level Reduction with Synchronization

```cpp
if (blockDim.x == 1024 && tid < 512) {
    i_data[tid] += i_data[tid + 512];
}
__syncthreads();

if (blockDim.x == 512 && tid < 256) {
    i_data[tid] += i_data[tid + 256];
}
__syncthreads();

if (blockDim.x == 256 && tid < 128) {
    i_data[tid] += i_data[tid + 128];
}
__syncthreads();

if (blockDim.x == 128 && tid < 64) {
    i_data[tid] += i_data[tid + 64];
}
__syncthreads();
```
- Each step halves the active threads by reducing pairs of elements.
- Synchronization ensures all threads complete their computation before proceeding.

### Warp-Level Reduction (Last 32 Threads)

```cpp
if (tid < 32) {
    volatile int* vsmem = i_data;

    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
}
```
- Uses `volatile` to ensure memory updates are visible across all threads in the warp.
- Explicitly performs reductions for offsets `32, 16, 8, 4, 2, 1` without synchronization.

### Final Block Result Storage

```cpp
if (tid == 0) {
    temp_array[blockIdx.x] = i_data[0];
}
```
- The first thread stores the block’s final reduced value in the `temp_array`.

---

## Benefits of Complete Unrolling

1. **Eliminates Loop Overhead**:
   - No conditional checks or increments in a loop structure.

2. **Reduces Synchronization**:
   - For the last warp, no `__syncthreads` is required, as all threads within a warp are inherently synchronized.

3. **Improves Parallel Efficiency**:
   - Fully utilizes the parallelism within a block, especially for large arrays.

4. **Scalable to Block Sizes**:
   - Handles block sizes of 1024, 512, 256, and 128 efficiently.

---

## Visualization of Execution Flow

### Test Case:
- Array Size: 2048 elements.
- Block Size: 1024 threads.

### Initial State:
- Input Array:
  ```
  [ A, B, C, D, ..., Z ]
  Threads: T0, T1, ..., T1023
  ```

### Unrolled Reduction Steps:

#### Block-Wide Reduction:
1. **Offset 512**:
   ```
   T0: A += I
   T1: B += J
   ...
   T511: H += P
   ```

2. **Offset 256**:
   ```
   T0: A += E
   T1: B += F
   ...
   T255: D += H
   ```

3. **Offset 128**:
   ```
   T0: A += C
   T1: B += D
   ...
   T127: Z += B
   ```

4. **Offset 64**:
   ```
   T0: A += B
   T1: C += D
   ...
   T63: Y += Z
   ```

#### Warp-Level Reduction:
- Threads `0–31` perform reductions directly:
  ```
  T0: A += F
  T1: B += G
  ...
  T15: O += P
  ```

### Final State:
- Block Sum stored in `temp_array[blockIdx.x]`.

---

## Final Aggregation on Host

The host combines all block results stored in `temp_array`:

```cpp
int gpu_result = 0;
for (int i = 0; i < gridDim.x; i++) {
    gpu_result += temp_array[i];
}
```

---

## Summary

Complete warp unrolling is an efficient approach to perform reductions in CUDA, especially for large arrays. By eliminating loops and minimizing synchronization, it achieves higher performance and scalability. The explicit handling of the last warp ensures warp efficiency, making this technique suitable for high-performance parallel reductions.

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

template<unsigned int iblock_size>
__global__ void reduction_kernel_warp_complete_unrolling(int* int_array, int* temp_array, int size) {
    int tid = threadIdx.x; // Thread index within the block

    // element index for this thread
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // local data pointer
    int* i_data = int_array + blockDim.x * blockIdx.x;

    // Complete unrolling of the reduction loop with synchronization after each step
    if (iblock_size == 1024 && tid < 512) {
        i_data[tid] += i_data[tid + 512];
    }
    __syncthreads();

    if (iblock_size == 512 && tid < 256) {
        i_data[tid] += i_data[tid + 256];
    }
    __syncthreads();

    if (iblock_size == 256 && tid < 128) {
        i_data[tid] += i_data[tid + 128];
    }
    __syncthreads();

    if (iblock_size == 128 && tid < 64) {
        i_data[tid] += i_data[tid + 64];
    }
    __syncthreads();

    // Handle the last warp (tid < 32) using volatile memory for direct updates
    if (tid < 32) {
        volatile int* vsmem = i_data; // Use volatile to prevent caching of memory operations

        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // Store the block's final result to the temp array
    if (tid == 0) {
        temp_array[blockIdx.x] = i_data[0];
    }
}

int main(int argc, char** argv) {
    // main
}
```

-----

# Dynamic Parallelism

**CUDA Dynamic Parallelism** allows new GPU kernels to be created and synchronized directly on the GPU.

## Features of Dynamic Parallelism
1. postpone the creation of number of blocks and grids until runtime
2. more transparent and easier recursive algorithm
3. reduce the need to transfer execution control and data between host and device

**Parent Grid:** started by the host thread

**Child Grid:** started by the parent grid

**Note:**
  - Grid launches are visible across a thread block
  - Execution of a thread block is not considered complete until all child grids created by all threads in the block have completed
  - When a parent launches a child grid, the child is not guranteed to begin execution until the parent thread block explicitly sync on the child
  - parent and child share the same global and constant memory storage, but they have distinct local and shared memory
  - there are two points in the execution of a child grid when its view of memory is fully consistent with the parent thread
    * at the start of a chiold grid
    * when child grid completes
  - shared and local memory are private to a thread block or thread, respectively, and are not visible or coherent between parent and child

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void dynamic_parallelism_check(int size, int depth) {
    printf(
        "Depth: %d - tid: %d \n",
        depth,
        threadIdx.x
    );

    if (size == 1) {
        return;
    }

    if (threadIdx.x == 0) {
        dynamic_parallelism_check<<<1, size / 2>>>(size / 2, depth + 1);
        cudaDeviceSynchronize();
    }
}

int main(int argc, char** argv) {
    dynamic_parallelism_check<<<1, 16>>>(16, 0);

    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
```

# To compile the above program

```
nvcc -arch=[your compute capability] -rdc=true <your program>.cu -o <output file>
```

-----

# Reduction Algorithm with Dynamic Parallelism

# Study Notes: Recursive Reduction Using CUDA Dynamic Parallelism

This document explains the recursive reduction algorithm implemented using CUDA dynamic parallelism. It highlights the key steps, components, and advantages of this approach.

---

## What is Recursive Reduction with Dynamic Parallelism?

Recursive reduction is an algorithm that repeatedly reduces an array of elements into smaller arrays until a single result is obtained. CUDA dynamic parallelism allows a kernel to launch other kernels, making recursion possible on the GPU. This method is particularly useful for hierarchical parallel reductions.

### Advantages of Dynamic Parallelism
1. **Simplified Host Code**:
   - The recursion is handled entirely on the GPU, reducing host intervention.
2. **Efficient Use of Resources**:
   - Each level of recursion processes fewer elements, making efficient use of threads.
3. **Hierarchical Parallelism**:
   - Allows finer control over thread and block configurations at each recursion level.

---

## Key Components of the Algorithm

### 1. Base Case

When the size of the array to be reduced (`isize`) reaches 2:

```cpp
if (isize == 2 && tid == 0) {
    g_odata[blockIdx.x] = idata[0] + idata[1];
    return;
}
```
- The first thread computes the sum of the two elements and writes the result to the output array.

### 2. Reduction Within a Block

At each recursion level, the threads within a block reduce the array in pairs:

```cpp
int istride = isize >> 1; // Compute stride (size / 2)
if (istride > 1 && tid < istride) {
    idata[tid] += idata[tid + istride];
}
__syncthreads();
```
- Threads add elements separated by the stride (`istride`).
- Synchronization ensures all threads complete their reduction step.

### 3. Recursive Kernel Launch

After performing block-level reduction, the first thread launches a new kernel to further reduce the array:

```cpp
if (tid == 0) {
    gpuRecursiveReduce<<<1, istride>>>(idata, odata, istride);
    cudaDeviceSynchronize();
}
```
- The kernel launches itself with a smaller array size and fewer threads (`istride`).
- Synchronization ensures the recursion completes before proceeding.

### 4. Final Result

The result of each block’s reduction is stored in `g_odata`. The host aggregates these partial results to compute the final sum.

---

## Execution Flow

### 1. Initial Setup (Host Code)
- Allocate and initialize arrays on the host and device.
- Launch the recursive kernel with the full array size and grid configuration.

### 2. Recursive Reduction (Device Code)
- **Block-Level Reduction**: Threads within a block reduce their portion of the array.
- **Recursive Kernel Launch**: The first thread of the block launches the next level of recursion with fewer threads and smaller arrays.

### 3. Host Aggregation
- The host aggregates the results from `g_odata` to compute the final sum.

---

## Visualization of Recursive Execution

### Example: Array Size = 16, Block Size = 8

#### Level 1:
- Input: `[A, B, C, D, E, F, G, H]`
- Threads: `T0, T1, ..., T7`
- Reduction:
  ```
  T0: A += E
  T1: B += F
  T2: C += G
  T3: D += H
  ```
- Result: `[A+E, B+F, C+G, D+H]`
- Recursive Launch:
  ```
  gpuRecursiveReduce<<<1, 4>>>(...) // 4 elements
  ```

#### Level 2:
- Input: `[A+E, B+F, C+G, D+H]`
- Threads: `T0, T1, T2, T3`
- Reduction:
  ```
  T0: (A+E) += (C+G)
  T1: (B+F) += (D+H)
  ```
- Result: `[A+E+C+G, B+F+D+H]`
- Recursive Launch:
  ```
  gpuRecursiveReduce<<<1, 2>>>(...) // 2 elements
  ```

#### Level 3:
- Input: `[A+E+C+G, B+F+D+H]`
- Threads: `T0, T1`
- Reduction:
  ```
  T0: (A+E+C+G) += (B+F+D+H)
  ```
- Final Result: `[Total Sum]`

---

## Key Points to Note

1. **Synchronization**:
   - `__syncthreads` ensures correctness during block-level reduction.
   - `cudaDeviceSynchronize` ensures the recursive kernel completes before proceeding.

2. **Scalability**:
   - Dynamic parallelism allows recursion to scale automatically based on the array size.

3. **Host Involvement**:
   - The host is only responsible for the initial kernel launch and final aggregation, simplifying the overall workflow.

---

## Comparison with Standard Reduction

| Aspect                 | Standard Reduction                    | Recursive Reduction                    |
|------------------------|----------------------------------------|----------------------------------------|
| **Host Involvement**   | Host launches multiple kernels         | Host launches one kernel               |
| **Synchronization**    | Explicit synchronization at each step | Implicit via dynamic parallelism       |
| **Scalability**        | Limited by fixed grid/block size       | Scales dynamically with array size     |

---

## Conclusion

The recursive reduction algorithm using CUDA dynamic parallelism simplifies the reduction process by leveraging the GPU’s capability to launch kernels recursively. It reduces host intervention and provides a scalable, hierarchical approach to parallel reduction. This method is especially effective for large datasets where multiple levels of reduction are required.

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void gpuRecursiveReduce(int* g_idata, int* g_odata, unsigned int isize) {
    int tid = threadIdx.x; // Thread index within the block

    // Pointers for the input and output data for this block
    int* idata = g_idata + blockIdx.x * blockDim.x;
    int* odata = &g_odata[blockIdx.x];

    // Base case: If the size of the array to be reduced is 2, perform the reduction directly
    if (isize == 2 && tid == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1]; // Add the two elements
        return;
    }

    // Compute the stride for reduction
    int istride = isize >> 1; // Divide the size by 2

    // Perform reduction within the current block
    if (istride > 1 && tid < istride) {
        idata[tid] += idata[tid + istride]; // Add pairs of elements
    }

    __syncthreads(); // Ensure all threads finish the reduction step

    // Launch the next level of recursion from the first thread of the block
    if (tid == 0) {
        gpuRecursiveReduce<<<1, istride>>>(idata, odata, istride); // Recursive kernel launch
        cudaDeviceSynchronize(); // Wait for the recursive kernel to finish
    }

    __syncthreads(); // Synchronize threads before returning
}

int main(int argc, char** argv) {
    const int size = 1 << 10; // Number of elements in the array (1024 elements)
    const int byte_size = size * sizeof(int);
    const int block_size = 128; // Threads per block

    int* h_input; // Host input array
    int* h_output; // Host output array (single element for final reduction result)

    // Allocate and initialize host memory
    h_input = (int*)malloc(byte_size);
    for (int i = 0; i < size; i++) {
        h_input[i] = rand() % 100; // Fill with random numbers
    }
    h_output = (int*)malloc(sizeof(int));

    // Allocate device memory
    int* d_input;
    int* d_output;
    cudaMalloc((void**)&d_input, byte_size);
    cudaMalloc((void**)&d_output, sizeof(int));

    // Copy input data to the device
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    // Calculate grid size based on block size
    dim3 block(block_size);
    dim3 grid((size + block.x - 1) / block.x);

    // Launch the recursive reduction kernel
    gpuRecursiveReduce<<<grid, block>>>(d_input, d_output, size);

    // Wait for GPU computations to complete
    cudaDeviceSynchronize();

    // Copy the final result back to the host
    cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // Verify the result on the CPU
    int cpu_result = 0;
    for (int i = 0; i < size; i++) {
        cpu_result += h_input[i];
    }

    // Print results
    printf("GPU Result: %d\n", *h_output);
    printf("CPU Result: %d\n", cpu_result);
    if (*h_output == cpu_result) {
        printf("Results match!\n");
    } else {
        printf("Results do NOT match!\n");
    }

    // Free host and device memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
```