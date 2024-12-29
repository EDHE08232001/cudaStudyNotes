# CUDA Example
```cpp
#include <cstdio>
#include <iostream>

using namespace std;

__global__ void maxi(int* a, int* b, int n)
{
    int block = 256 * blockIdx.x;
    int max = 0;

    for (int i = block; i < min(256 + block, n); i++) {

        if (max < a[i]) {
            max = a[i];
        }
    }
    b[blockIdx.x] = max;
}

int main()
{

    int n;
    n = 3 >> 2;
    int a[n];

    for (int i = 0; i < n; i++) {
        a[i] = rand() % n;
        cout << a[i] << "\t";
    }

    cudaEvent_t start, end;
    int *ad, *bd;
    int size = n * sizeof(int);
    cudaMalloc(&ad, size);
    cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
    int grids = ceil(n * 1.0f / 256.0f);
    cudaMalloc(&bd, grids * sizeof(int));

    dim3 grid(grids, 1);
    dim3 block(1, 1);

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    while (n > 1) {
        maxi<<<grids, block>>>(ad, bd, n);
        n = ceil(n * 1.0f / 256.0f);
        cudaMemcpy(ad, bd, n * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float time = 0;
    cudaEventElapsedTime(&time, start, end);

    int ans[2];
    cudaMemcpy(ans, ad, 4, cudaMemcpyDeviceToHost);

    cout << "The maximum element is : " << ans[0] << endl;

    cout << "The time required : ";
    cout << time << endl;
}
```

-----

# **Threads, Blocks, and Grids**

CUDA uses a hierarchy to organize and manage parallel execution on a GPT:
1.   Threads
2.   Blocks
3.   Grids

**Threads:**
*   Smallest unit of execution in CUDA. Each thread executes the same code (kernel), but operates on different data.
*   each thread is identified by using *threadIdx*

**Blocks:**
* A group of threads that execute together and can communicate via shared memory. Each block runs independently of other blocks
* Identified by *blockIdx*

**Grids:**
* A collection of blocks that are launched to execute a kernel. The entire grid executes the kernel fucntion on the GPU

---

# **Dimensions in CUDA**
Thread, blocks, and grids can be structured in a 1D, 2D, or 3D layout. These layouts are defined using:
* **`dim3`:** A built-in CUDA type used to specify dimensions in each kernel launch. Using it, you can set dimensions for blocks (blockDim) and grids (gridDim).

## threadIdx
* x
* y
* z

*Note:* Each dimension identifies the thread's position along each respective dimension within its block

Example:
A grid with `blockDim.x = 4`, `blockDim.y = 3`, and `blockDim.z = 2`, threads will be organized in a 4x3x2 structure

## blockIdx
* x
* y
* z

*Note:* Each dimension identifies the block's position along each respective dimension within the grid

Example:
A grid with `gridDim.x = 3` and `gridDim.y = 2` has 3x2 structure

## blockDim
* x
* y
* z

*Note:* each dimension is the total number of threads in the each respective dimention of a block

---

# **<<<>>>** in kernel function calls
**`kernelFunciton<<<gridDim, blockDim, sharedMemSize, stream>>>(kernelArguments)`**

* `gridDim`
  - how many blocks are in the grid along each dimension
  - Typically dfined as `dim3(gridDim.x, gridDim.y, gridDim.z)`
* `blockDim`
  - how many threads are in each block along each dimension
  - Typically defined as `dim3(blockDim.x, blockDim.y, blockDim.z)`
* `sharedMemSize`
  - Options, if not specified then 0
  - amount of dynamic shared memory (in bytes) that each block can use
* `stream`
  - optional
  - specifies the stream in which the kernel should be executed

---

# Default Behavior

```cuda
#include <iostream>

__global__ void kernelExample() {
  // Kernel Code
}

int main() {
  // Launch the kernel without specifying x, y, z
  dim3 gridDim; // Defaults to (1, 1, 1)
  dim3 blockDim; // Defaults to (1, 1, 1)

  // Kernel Launch
  kernelExample<<<dridDim, blockDim>>>();
  cudaDeviceSynchronize();

  return 0;
}
```
In `<<<x, y, sharedMem, stream>>>`
- x means gridDim(x, 1, 1)
- y means blockDim(y, 1, 1)

---

# To specify x, y, and z

```cuda
dim3 gridDim(x, y, z);
dim3 blockDim(x, y, z);
```

---

# `cudaDeviceSynchronize()`
* block the host (CPU) until all previously issued commands to the GPU (Device) have been completed
* When to use?
  - Error Checking
  - Ensuring correct sequencing
  - Performance Measurements
* Behavior
  - block the CPU until the GPU has completed all pending series of GPU operations

-----

# **Basic Steps of a CUDA Program**
* Initialization of data from CPU
* Transfer data from CPU context to GPU context
* Kernel launch with needed grid/block size
* Transfer results back to CPU context from GPU context
* Reclaim the memory from both CPU and GPU

-----

# Global Index Calculation
- `int globalIndexX = threadIdx.x + blockDim.x * blockIdx.x` (In 1D config)
- `int globalIndexY = threadIdx.y + blockDim.y * blockIdx.y` (With 2D config)
- `int globalIndexZ = threadIdx.z + blockDim.z * blockIdx.z` (With 3D config)

-----

# Limitations for Block Size
* x ≤ 1024
* y ≤ 1024
* z ≤ 64

*Note:* x*y*z ≤ 1024

---

# Limitation for Number of Thread Block in Each Dimension
* x ≤ 2^32 - 1
* y ≤ 65536
* z ≤ 65536

-----

# Organization of Threads In CUDA

* threadIdx
  - CUDA runtime uniquely initialized threadIdx variable for each thread depending on where that particular thread is located in the thread block
  - threadIdx is dim3 type variable

-----

# **blockIdx**
CUDA runtime uniquely initialized blockIdx variable for each thread depending on the coordinates of the belonging thread block in the grid

*Note:* `blockIdx` is a dim3 type variable

-----

# **blockDim**
`blockDim` variable consists number of threads in each dimension of a thread block. Notice all the thread blocks in a grid have same block size, so this variable is same for all the threads in a grid

*Note:* `blockDim` is a dim3 type variable

-----

# **gridDim**
consists number of thread blocks in each dimension of a grid

*Note:* dim3 type variable

-----

# Unique Index Calculation
In CUDA, it is very common to use `threadIdx`, `blockIdx`, `blockDim`, `gridDim` variable values to calculate array indices.

```cpp
#include <stdio.h>
#include <stdlib.h>

// Kernel function to calculate globally unique index across all threads
__global__ void unique__gid__calculation(int* input) {
    // tid is the thread index within the current block
    int tid = threadIdx.x;

    // blockIdx.x gives the index of the block, blockDim.x gives the size of each block
    // Calculate the offset by multiplying block index by block size
    int offset = blockIdx.x * blockDim.x;

    // Global index (gid) is calculated by adding the thread index (tid) to the offset
    int gid = tid + offset; // calculating global index

    // Print the block index, thread index, global index, and the input value at that global index
    printf(
        "blockIdx.x: %d, threadIdx.x: %d, gid: %d, value: %d \n",
        blockIdx.x,
        threadIdx.x,
        gid,
        input[gid]
    );
}

int main() {
    // Define the size of the array (number of elements)
    int array_size = 16;

    // Calculate the total byte size of the array (size of an int * number of elements)
    int array_byte_size = sizeof(int) * array_size;

    // Host array containing 16 integers
    int h_data[] = {23, 9, 4, 53, 65, 12, 1, 33, 87, 45, 23, 12, 342, 56, 44, 99};

    // Print the host array data
    for (int i = 0; i < array_size; i++) {
        printf("%d ", h_data[i]);
    }
    printf("\n \n");

    // Device pointer to hold the array data in GPU memory
    int* d_data;

    // Allocate memory on the GPU for the array
    cudaMalloc((void**)&d_data, array_byte_size);

    // Copy the array from host (CPU) to device (GPU)
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    // Define the block size (number of threads per block)
    // Here, we are launching 4 threads per block
    dim3 block(4);

    // Define the grid size (number of blocks per grid)
    // We are launching 4 blocks in total
    dim3 grid(4);

    // Visualizing the grid and block layout:
    /*
     * Grid has 4 blocks, and each block has 4 threads:
     *
     * Block 0 (blockIdx.x = 0):
     * threadIdx.x = 0 -> gid = 0
     * threadIdx.x = 1 -> gid = 1
     * threadIdx.x = 2 -> gid = 2
     * threadIdx.x = 3 -> gid = 3
     *
     * Block 1 (blockIdx.x = 1):
     * threadIdx.x = 0 -> gid = 4
     * threadIdx.x = 1 -> gid = 5
     * threadIdx.x = 2 -> gid = 6
     * threadIdx.x = 3 -> gid = 7
     *
     * Block 2 (blockIdx.x = 2):
     * threadIdx.x = 0 -> gid = 8
     * threadIdx.x = 1 -> gid = 9
     * threadIdx.x = 2 -> gid = 10
     * threadIdx.x = 3 -> gid = 11
     *
     * Block 3 (blockIdx.x = 3):
     * threadIdx.x = 0 -> gid = 12
     * threadIdx.x = 1 -> gid = 13
     * threadIdx.x = 2 -> gid = 14
     * threadIdx.x = 3 -> gid = 15
     *
     * As we can see, the gid (global index) ranges from 0 to 15 across the 4 blocks and 4 threads.
     */

    // Launch the kernel with the specified grid and block dimensions
    // Each block will have 4 threads, and the grid will have 4 blocks
    // Total threads = grid.x * block.x = 4 * 4 = 16 threads
    unique__gid__calculation<<<grid, block>>>(d_data);

    // Wait for the GPU to finish executing the kernel
    cudaDeviceSynchronize();

    // Reset the device (GPU)
    cudaDeviceReset();

    return 0;
}
```

-----

# Unique Index Calculation for 2D-Grid
**General Equation:** Index = row offset + block offset + tid

```cpp
#include <stdio.h>
#include <stdlib.h>

// Kernel function to calculate globally unique index across a 2D grid of blocks
__global__ void unique__gid__calculation_2d(int* input) {
    // tid is the thread index within the current block (in the x-dimension)
    int tid = threadIdx.x;

    // Calculate the row offset by multiplying the number of blocks in the x-dimension (gridDim.x)
    // by the block size (blockDim.x), and then multiplying by blockIdx.y (the block's y index)
    int row_offset = gridDim.x * blockDim.x * blockIdx.y; // row offset for the y-dimension

    // Calculate the block offset within the row by multiplying the block index in x by block size
    int block_offset = blockIdx.x * blockDim.x; // block offset within the x-dimension

    // Calculate the global index by adding the thread index (tid), row offset, and block offset
    int gid = tid + row_offset + block_offset; // calculating global index

    // Print the block indices, thread index, global index, and the input value at that global index
    printf(
        "blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, gid: %d, value: %d \n",
        blockIdx.x,
        blockIdx.y,
        tid,
        gid,
        input[gid]
    );
}

int main() {
    // Define the size of the array (number of elements)
    int array_size = 16;

    // Calculate the total byte size of the array (size of an int * number of elements)
    int array_byte_size = sizeof(int) * array_size;

    // Host array containing 16 integers
    int h_data[] = {23, 9, 4, 53, 65, 12, 1, 33, 87, 45, 23, 12, 342, 56, 44, 99};

    // Print the host array data
    for (int i = 0; i < array_size; i++) {
        printf("%d ", h_data[i]);
    }
    printf("\n \n");

    // Device pointer to hold the array data in GPU memory
    int* d_data;

    // Allocate memory on the GPU for the array
    cudaMalloc((void**)&d_data, array_byte_size);

    // Copy the array from host (CPU) to device (GPU)
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    // Define the block size (number of threads per block)
    // Here, we are launching 4 threads per block
    dim3 block(4);

    // Define the grid size (2x2 grid, so 4 blocks in total)
    // This means the grid has 2 blocks in the x-dimension and 2 blocks in the y-dimension
    dim3 grid(2, 2);

    // Visualizing the grid and block layout:
    /*
     * Grid has 2 blocks in x (gridDim.x = 2) and 2 blocks in y (gridDim.y = 2).
     * Each block has 4 threads.
     *
     * The global index (gid) is calculated as:
     * gid = threadIdx.x + (gridDim.x * blockDim.x * blockIdx.y) + (blockIdx.x * blockDim.x)
     *
     * Breakdown of grid and global indices:
     *
     * Block (0,0):  blockIdx.x = 0, blockIdx.y = 0
     * threadIdx.x = 0 -> gid = 0
     * threadIdx.x = 1 -> gid = 1
     * threadIdx.x = 2 -> gid = 2
     * threadIdx.x = 3 -> gid = 3
     *
     * Block (1,0):  blockIdx.x = 1, blockIdx.y = 0
     * threadIdx.x = 0 -> gid = 4
     * threadIdx.x = 1 -> gid = 5
     * threadIdx.x = 2 -> gid = 6
     * threadIdx.x = 3 -> gid = 7
     *
     * Block (0,1):  blockIdx.x = 0, blockIdx.y = 1
     * threadIdx.x = 0 -> gid = 8
     * threadIdx.x = 1 -> gid = 9
     * threadIdx.x = 2 -> gid = 10
     * threadIdx.x = 3 -> gid = 11
     *
     * Block (1,1):  blockIdx.x = 1, blockIdx.y = 1
     * threadIdx.x = 0 -> gid = 12
     * threadIdx.x = 1 -> gid = 13
     * threadIdx.x = 2 -> gid = 14
     * threadIdx.x = 3 -> gid = 15
     *
     * As we can see, the gid (global index) ranges from 0 to 15 across the 2D grid of 4 blocks (2x2) and 4 threads per block.
     */

    // Launch the kernel with the specified grid and block dimensions
    // Grid of 2x2 blocks, with each block containing 4 threads
    unique__gid__calculation_2d<<<grid, block>>>(d_data);

    // Wait for the GPU to finish executing the kernel
    cudaDeviceSynchronize();

    // Reset the device (GPU)
    cudaDeviceReset();

    return 0;
}
```

-----

# Scenario
0 1 | 4 5

2 3 | 6 7

...

## The memory access pattern is going to depend on the way we calculate our global index

## Usually, we prefer to calculate goobal indices in a way that, threads with in same thread block access consecutive memory locations or consecutive elements in the array

## `tid = threadIdx.y * blockDim.x + threadIdx.x`

## block_offset = numberThreadInBlock * blockIdx.x
  - and in code it is
  - `block_offset = (blockDim.x * blockDim.y) * blockIdx.x`

## row_offset = number of trheads in a row * blockIdx.y
  - in code
  - `row_offset = (blockDim.x * blockDim.y * gridDim.x) * blockIdx.y`

```cpp
#include <stdio.h>
#include <stdlib.h>

// Kernel function to calculate globally unique index across a 2D grid of 2D blocks
__global__ void unique_gid_calculation_2d_2d(int *data) {
    // Calculate the thread index (tid) within the 2D block
    // tid = threadIdx.y * blockDim.x + threadIdx.x
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    // Calculate the total number of threads in a block
    int num_threads_in_a_block = blockDim.x * blockDim.y;

    // Calculate the block offset within the current row (x-dimension)
    int block_offset = blockIdx.x * num_threads_in_a_block;

    // Calculate the total number of threads in a single row of blocks
    // This is the number of threads in one block multiplied by the number of blocks in a row (gridDim.x)
    int num_threads_in_a_row = num_threads_in_a_block * gridDim.x;

    // Calculate the row offset (y-dimension) by multiplying the number of threads in a row by blockIdx.y
    int row_offset = num_threads_in_a_row * blockIdx.y;

    // Calculate the global index (gid) as the sum of tid, block_offset, and row_offset
    int gid = tid + block_offset + row_offset;

    // Print the block indices, thread index (tid), global index (gid), and the corresponding data value
    printf(
        "blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, tid: %d, gid: %d - data: %d \n",
        blockIdx.x,
        blockIdx.y,
        threadIdx.x,
        tid,
        gid,
        data[gid]
    );
}

int main() {
    // Define the size of the array (number of elements)
    int array_size = 16;

    // Calculate the total byte size of the array (size of an int * number of elements)
    int array_byte_size = sizeof(int) * array_size;

    // Host array containing 16 integers
    int h_data[] = {23, 9, 4, 53, 65, 12, 1, 33, 22, 43, 56, 4, 76, 81, 94, 32};

    // Device pointer to hold the array data in GPU memory
    int *d_data;

    // Allocate memory on the GPU for the array
    cudaMalloc((void**)&d_data, array_byte_size);

    // Copy the array from host (CPU) to device (GPU)
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    // Define the block size as 2x2 (2 threads in x, 2 threads in y), so each block has 4 threads
    dim3 block(2, 2);

    // Define the grid size as 2x2, meaning the grid has 2 blocks in x-dimension and 2 blocks in y-dimension
    dim3 grid(2, 2);

    // Visualizing the grid and block layout:
    /*
     * Grid has 2 blocks in x (gridDim.x = 2) and 2 blocks in y (gridDim.y = 2).
     * Each block is 2x2, meaning 4 threads per block.
     *
     * The global index (gid) is calculated as:
     * gid = threadIdx.y * blockDim.x + threadIdx.x + (blockIdx.x * blockDim.x * blockDim.y) + (blockIdx.y * blockDim.x * blockDim.y * gridDim.x)
     *
     * Breakdown of grid and global indices:
     *
     * Block (0,0):  blockIdx.x = 0, blockIdx.y = 0
     * threadIdx = (0,0) -> gid = 0
     * threadIdx = (0,1) -> gid = 1
     * threadIdx = (1,0) -> gid = 2
     * threadIdx = (1,1) -> gid = 3
     *
     * Block (1,0):  blockIdx.x = 1, blockIdx.y = 0
     * threadIdx = (0,0) -> gid = 4
     * threadIdx = (0,1) -> gid = 5
     * threadIdx = (1,0) -> gid = 6
     * threadIdx = (1,1) -> gid = 7
     *
     * Block (0,1):  blockIdx.x = 0, blockIdx.y = 1
     * threadIdx = (0,0) -> gid = 8
     * threadIdx = (0,1) -> gid = 9
     * threadIdx = (1,0) -> gid = 10
     * threadIdx = (1,1) -> gid = 11
     *
     * Block (1,1):  blockIdx.x = 1, blockIdx.y = 1
     * threadIdx = (0,0) -> gid = 12
     * threadIdx = (0,1) -> gid = 13
     * threadIdx = (1,0) -> gid = 14
     * threadIdx = (1,1) -> gid = 15
     *
     * As we can see, the gid (global index) ranges from 0 to 15 across the 2D grid of 2x2 blocks, each containing 2x2 threads.
     */

    // Launch the kernel with the specified grid and block dimensions
    // Grid of 2x2 blocks, with each block containing 2x2 threads
    unique_gid_calculation_2d_2d<<<grid, block>>>(d_data);

    // Wait for the GPU to finish executing the kernel
    cudaDeviceSynchronize();

    // Reset the device (GPU)
    cudaDeviceReset();

    return 0;
}
```

-----

# CUDA Memory Transfer

## Visual Diagram
```
Initialize Data
      ⬇️    ---------------------->      Device Execution
   Host Logic                                   |
      ⬇️                                        |
Waiting for the GPU Results   <-----------------|
```

## Transfer memory between host(cpu) and device(gpu) using `cudaMemCpy()` function
```
cudaMemCpy(
  destination_ptr,
  source_ptr,
  size_in_byte,
  direction
);
```
### Direction
  - Host to device - cudamemcpyhtod
  - Device to host - cudamemcpydtoh
  - Device to device - cudamemcpydtod

```cpp
#include <stdio.h>        // Required for standard input/output functions (e.g., printf)
#include <stdlib.h>       // Required for memory allocation and random number generation (e.g., malloc, rand)
#include <time.h>         // Required to seed the random number generator using the current time
#include <cuda_runtime.h> // Required for CUDA functions such as cudaMalloc, cudaMemcpy, and cudaDeviceSynchronize

// Kernel function that runs on the GPU to print thread and block information
__global__ void mem_trs_test(int *input) {
    // Calculate the global index (gid) of each thread.
    // Global ID is calculated by combining the block index and thread index.
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Print the thread index (tid), global index (gid), and the corresponding value from the input array
    printf("tid: %d, gid: %d, value: %d \n", threadIdx.x, gid, input[gid]);
}

// Modified kernel function that checks the size before printing
__global__ void mem_trs_test_with_size_restriction(int *input, int size) {
    // Calculate the global index (gid) of each thread
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure that gid is within the size of the array to prevent out-of-bounds access
    if (gid < size) {
        // Print the thread index (tid), global index (gid), and the corresponding value from the input array
        printf("tid: %d, gid: %d, value: %d \n", threadIdx.x, gid, input[gid]);
    }
}

int main() {
    // Define the number of elements in the array
    int size = 150;  // We are working with an array of 150 integers

    // Calculate the total size of the array in bytes (each int takes 4 bytes)
    int byte_size = size * sizeof(int);

    // Allocate memory on the host (CPU)
    int *h_input;
    h_input = (int*)malloc(byte_size);  // Dynamically allocate memory for the input array on the host (CPU)

    // Initialize random seed based on the current time
    time_t t;
    srand((unsigned)time(&t));  // Seed the random number generator with the current time to generate different random numbers each run

    // Fill the host array with random integers between 0 and 255
    for (int i = 0; i < size; i++) {
        h_input[i] = (int)(rand() & 0xff);  // Mask the random value to get an 8-bit number (0 to 255)
    }

    // Declare a pointer for the device (GPU) array
    int *d_input;

    // Allocate memory on the GPU (device) for the input array
    // cudaMalloc allocates 'byte_size' bytes of memory on the GPU, and the device pointer d_input will point to that memory
    cudaMalloc((void**)&d_input, byte_size);

    // Copy data from the host (CPU) to the device (GPU)
    // cudaMemcpy transfers data between host and device. Here, we're copying 'byte_size' bytes from the host array (h_input) to the device array (d_input)
    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    // Set up the block and grid dimensions for the kernel launch
    // Block size is set to 32 threads per block. This means each block will run 32 threads.
    dim3 block(32);  // Set block dimensions (32 threads in the x direction)

    // Grid size is set to 5 blocks. This means we will have 5 blocks, each containing 32 threads, for a total of 160 threads.
    dim3 grid(5);  // Set grid dimensions (5 blocks in the x direction)

    // Launch the kernel on the GPU
    // The kernel will run with 5 blocks, each containing 32 threads. This means a total of 160 threads will be launched.
    mem_trs_test<<<grid, block>>>(d_input);

    // Synchronize the CPU and GPU to ensure the kernel execution is complete before proceeding
    // cudaDeviceSynchronize ensures that the CPU waits for the GPU to finish before moving on. Without this, the program may continue execution without waiting for the kernel to finish.
    cudaDeviceSynchronize();

    // Free the allocated memory on the GPU
    // Always free the memory on the GPU that was allocated with cudaMalloc to prevent memory leaks on the device
    cudaFree(d_input);

    // Free the allocated memory on the host
    // Free the dynamically allocated memory on the host (CPU)
    free(h_input);

    // Reset the device (GPU) to its original state
    // Resetting the GPU clears its state and ensures there are no leftover data or settings from previous runs
    cudaDeviceReset();

    return 0;  // Return 0 to indicate successful execution
}
```

-----

# Error Handling In Cuda Program: With Array Sum Example

## Types of Errors
  - Compile Time Errors
    * Errors due to not adhering to language syntax
  - Runtime Errors
    * Happens while program is running

In c++ you can handle errors with
```
try {
  // code
} catch (ExceptionClass &ex) {
  // handling
}
```

## Error Handling in CUDA
We are working with two separate hardwares. Error transfering is needed(for errors happened in device to host).

**For this purpose** every CUDA function other than kernel launches return cudaError.

`cudaError cuda_function(......)`


`cudaError` is an `enum`, that is defined in the CUDA API.
  - if returned values is `cudaSuccess`, operation is good
  - otherwise, values other than `cudaSuccess` is returned

Then we can pass returned `cudaError` to `cudaGetString(error)` function to get the text corresponding to the error code.

```cpp
#include <stdio.h>  // Standard input/output library for printing to the console

// Libraries for random initialization
#include <stdlib.h> // Standard library for memory allocation, random numbers, etc.
#include <time.h>  // Library for time-based functions (used for random number seeding)

// Library for memory operations like memset
#include <cstring>  // For memory functions like memset

// Define a macro for error checking in CUDA
#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__);}

// Helper function to check for errors returned by CUDA functions
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {  // Check if the CUDA function returned an error
        // Print the error message, file, and line number where the error occurred
        fprintf(stderr, "GPUassert: %s %s, %d \n", cudaGetErrorString(code), file, line);
        if (abort) {  // If the abort flag is true, exit the program with the error code
            exit(code);
        }
    }
}

// CUDA kernel function to perform element-wise addition of two arrays on the GPU
__global__ void sum_array_gpu(int* a, int* b, int* c, int size) {
    // Calculate the global index for each thread
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the global index is within bounds of the array size
    if (gid < size) {
        // Perform the addition at this index
        c[gid] = a[gid] + b[gid];
    }
}

// CPU function for array addition to compare results with the GPU
void sum_array_cpu(int* a, int* b, int* c, int size) {
    // Loop over each element and perform addition
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

// Function to compare two arrays and check if they are equal
void compare_arrays(int* a, int* b, int size) {
    // Loop through each element to compare arrays
    for (int i = 0; i < size; i++) {
        // If any element differs, print that the arrays are different
        if (a[i] != b[i]) {
            printf("Arrays are different \n");
            return;
        }
    }
    // If all elements match, print that the arrays are the same
    printf("Arrays are same! \n");
}

int main() {
    // Define the size of the arrays (10,000 elements in each)
    int size = 10000;
    int block_size = 128;  // CUDA block size (number of threads per block)
    cudaError error;  // Variable to store CUDA errors

    // Calculate the number of bytes needed for each array (size of int * number of elements)
    int NO_BYTES = size * sizeof(int);

    // Host pointers (CPU memory)
    int* h_a;           // Pointer for array A on the host
    int* h_b;           // Pointer for array B on the host
    int* gpu_results;   // Pointer for storing GPU results on the host
    int* h_c;           // Pointer for storing CPU results on the host

    // Allocate memory on the CPU for arrays A, B, and results
    h_a = (int*)malloc(NO_BYTES);  // Allocate memory for array A
    h_b = (int*)malloc(NO_BYTES);  // Allocate memory for array B
    gpu_results = (int*)malloc(NO_BYTES);  // Allocate memory for GPU results
    h_c = (int*)malloc(NO_BYTES);  // Allocate memory for CPU results

    // Initialize arrays with random values
    time_t t;
    srand((unsigned)time(&t));  // Seed the random number generator with the current time

    // Fill arrays A and B with random values (range 0-255)
    for (int i = 0; i < size; i++) {
        h_a[i] = (int)(rand() & 0xFF);  // Random values for array A
    }
    for (int i = 0; i < size; i++) {
        h_b[i] = (int)(rand() & 0xFF);  // Random values for array B
    }

    // Clear the CPU results array (set all elements to 0)
    memset(h_c, 0, NO_BYTES);

    // Clear the GPU results array (set all elements to 0)
    memset(gpu_results, 0, NO_BYTES);

    // Perform array addition on the CPU
    sum_array_cpu(h_a, h_b, h_c, size);  // Pass array size to the function

    // Device pointers (GPU memory)
    int* d_a;  // Pointer for array A on the GPU
    int* d_b;  // Pointer for array B on the GPU
    int* d_c;  // Pointer for result array on the GPU

    // Allocate memory on the GPU for arrays A, B, and C
    // Error checking is done using the gpuErrchk macro
    gpuErrchk(cudaMalloc((int**)&d_a, NO_BYTES));  // Allocate memory for array A on the GPU
    gpuErrchk(cudaMalloc((int**)&d_b, NO_BYTES));  // Allocate memory for array B on the GPU
    gpuErrchk(cudaMalloc((int**)&d_c, NO_BYTES));  // Allocate memory for result array C on the GPU

    // Copy arrays A and B from host to device (GPU memory)
    gpuErrchk(cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice));  // Copy array A to the GPU
    gpuErrchk(cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice));  // Copy array B to the GPU

    // Configure the execution grid (number of blocks and threads per block)
    dim3 block(block_size);  // Define block size (128 threads per block)
    dim3 grid((size / block.x) + 1);  // Define grid size (number of blocks)

    // Launch the GPU kernel to perform array addition
    sum_array_gpu<<<grid, block>>>(d_a, d_b, d_c, size);  // Launch the kernel on the GPU

    // Wait for the GPU to finish processing, and use error checking for CUDA synchronization
    gpuErrchk(cudaDeviceSynchronize());

    // Copy the result array C from the device (GPU) back to the host
    gpuErrchk(cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost));

    // Compare the results from the GPU and CPU
    compare_arrays(h_c, gpu_results, size);  // Compare both arrays

    // Free allocated memory on the GPU
    gpuErrchk(cudaFree(d_c));  // Free memory for the result array on the GPU
    gpuErrchk(cudaFree(d_b));  // Free memory for array B on the GPU
    gpuErrchk(cudaFree(d_a));  // Free memory for array A on the GPU

    // Free allocated memory on the CPU
    free(h_c); // Free memory for cpu results
    free(gpu_results);  // Free memory for GPU results
    free(h_a);  // Free memory for array A
    free(h_b);  // Free memory for array B

    return 0;  // Return successful execution
}
```

-----

# Performance of a CUDA Application
  - Execution Time
  - Power Consumption
  - Floor Space
  - Cost of Hardware

# Trail and Error Method
Running the CUDA program with different grid, block, shared memory, cache, memory access configurations and choose the best config based on the execution time.

-----

# CUDA Device Properties
  - Depending on CUDA device compute capability, properties of CUDA device is going to vary.
  - When we program a CUDA application to run on device with multiple compute capabilities, then we need a way to query the device properties on the fly.

```cpp
#include <stdio.h>

// Function to query and print properties of the CUDA device
void query_device() {
    int deviceCount = 0;

    // Get the number of CUDA devices available
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA support device found\n");
        return;
    }

    int devNo = 0;  // Select device 0 (the first device, assuming there's at least one)

    // Create a variable to hold the device properties
    cudaDeviceProp iProp;

    // Get properties of the selected device (devNo)
    cudaGetDeviceProperties(&iProp, devNo);  // This function populates iProp with the properties of device devNo

    // Print the name of the CUDA device
    printf("Device %d: %s \n", devNo, iProp.name);

    // Print the number of multiprocessors (SMs) on the device
    printf("Number of multiprocessors:                                      %d \n", iProp.multiProcessorCount);

    // Print the clock rate of the CUDA cores (in KHz)
    printf("Clock Rate:                                                     %d KHz\n", iProp.clockRate);

    // Print the compute capability (major and minor version)
    printf("Compute Capability:                                             %d.%d \n", iProp.major, iProp.minor);

    // Print the total global memory (converted to KB)
    printf("Total Amount of Global Memory:                                  %4.2f KB \n", iProp.totalGlobalMem / 1024.0);

    // Print the total amount of constant memory (converted to KB)
    printf("Total Amount of Constant Memory:                                %4.2f KB \n", iProp.totalConstMem / 1024.0);

    // Print the shared memory available per block (converted to KB)
    printf("Total Amount of Shared Memory Per Block:                        %4.2f KB \n", iProp.sharedMemPerBlock / 1024.0);

    // Additional device properties
    // Print the maximum number of threads per block
    printf("Maximum Number of Threads Per Block:                            %d \n", iProp.maxThreadsPerBlock);

    // Print the maximum dimensions of a block (x, y, z)
    printf("Maximum Block Dimensions:                                       %d x %d x %d \n",
           iProp.maxThreadsDim[0], iProp.maxThreadsDim[1], iProp.maxThreadsDim[2]);

    // Print the maximum dimensions of a grid (x, y, z)
    printf("Maximum Grid Dimensions:                                        %d x %d x %d \n",
           iProp.maxGridSize[0], iProp.maxGridSize[1], iProp.maxGridSize[2]);

    // Print the memory bus width (in bits)
    printf("Memory Bus Width:                                               %d bits\n", iProp.memoryBusWidth);

    // Print the peak memory clock rate (in KHz)
    printf("Memory Clock Rate:                                              %d KHz\n", iProp.memoryClockRate);

    // Print the L2 cache size (in KB)
    printf("L2 Cache Size:                                                  %d KB \n", iProp.l2CacheSize / 1024);

    // Print warp size (number of threads in a warp)
    printf("Warp Size:                                                      %d \n", iProp.warpSize);

    // Print the maximum amount of registers available per block
    printf("Maximum Number of Registers Per Block:                          %d \n", iProp.regsPerBlock);

    // Print whether the device supports unified addressing
    printf("Supports Unified Addressing:                                    %s \n", iProp.unifiedAddressing ? "Yes" : "No");

    // Print if the device supports ECC (Error-Correcting Code) memory
    printf("Supports ECC Memory:                                            %s \n", iProp.ECCEnabled ? "Yes" : "No");

    // Print the number of async engines (for concurrent copy and execution)
    printf("Number of Asynchronous Engines:                                 %d \n", iProp.asyncEngineCount);

    // Print the device's ability to overlap memory transfers and kernel execution
    printf("Device Can Overlap Memory Transfers and Kernel Execution:        %s \n", iProp.deviceOverlap ? "Yes" : "No");
}

int main() {
    // Query and print CUDA device properties
    query_device();
    return 0;
}
```

-----

# Print properties for all devices

```cpp
#include <stdio.h>

// Function to query and print properties of all CUDA devices
void query_all_devices() {
    int deviceCount = 0;

    // Get the number of CUDA devices available
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA support device found\n");
        return;
    }

    // Loop through each CUDA device and print its properties
    for (int devNo = 0; devNo < deviceCount; devNo++) {
        // Create a variable to hold the device properties
        cudaDeviceProp iProp;

        // Get properties of the selected device (devNo)
        cudaGetDeviceProperties(&iProp, devNo);  // This function populates iProp with the properties of device devNo

        // Print device number and name
        printf("Device %d: %s \n", devNo, iProp.name);

        // Print the number of multiprocessors (SMs) on the device
        printf("Number of multiprocessors:                                      %d \n", iProp.multiProcessorCount);

        // Print the clock rate of the CUDA cores (in KHz)
        printf("Clock Rate:                                                     %d KHz\n", iProp.clockRate);

        // Print the compute capability (major and minor version)
        printf("Compute Capability:                                             %d.%d \n", iProp.major, iProp.minor);

        // Print the total global memory (converted to KB)
        printf("Total Amount of Global Memory:                                  %4.2f KB \n", iProp.totalGlobalMem / 1024.0);

        // Print the total amount of constant memory (converted to KB)
        printf("Total Amount of Constant Memory:                                %4.2f KB \n", iProp.totalConstMem / 1024.0);

        // Print the shared memory available per block (converted to KB)
        printf("Total Amount of Shared Memory Per Block:                        %4.2f KB \n", iProp.sharedMemPerBlock / 1024.0);

        // Additional device properties
        printf("Maximum Number of Threads Per Block:                            %d \n", iProp.maxThreadsPerBlock);

        printf("Maximum Block Dimensions:                                       %d x %d x %d \n",
               iProp.maxThreadsDim[0], iProp.maxThreadsDim[1], iProp.maxThreadsDim[2]);

        printf("Maximum Grid Dimensions:                                        %d x %d x %d \n",
               iProp.maxGridSize[0], iProp.maxGridSize[1], iProp.maxGridSize[2]);

        printf("Memory Bus Width:                                               %d bits\n", iProp.memoryBusWidth);

        printf("Memory Clock Rate:                                              %d KHz\n", iProp.memoryClockRate);

        printf("L2 Cache Size:                                                  %d KB \n", iProp.l2CacheSize / 1024);

        printf("Warp Size:                                                      %d \n", iProp.warpSize);

        printf("Maximum Number of Registers Per Block:                          %d \n", iProp.regsPerBlock);

        printf("Supports Unified Addressing:                                    %s \n", iProp.unifiedAddressing ? "Yes" : "No");

        printf("Supports ECC Memory:                                            %s \n", iProp.ECCEnabled ? "Yes" : "No");

        printf("Number of Asynchronous Engines:                                 %d \n", iProp.asyncEngineCount);

        printf("Device Can Overlap Memory Transfers and Kernel Execution:        %s \n", iProp.deviceOverlap ? "Yes" : "No");

        printf("\n\n");  // Separate output for each device
    }
}

int main() {
    // Query and print CUDA device properties for all devices
    query_all_devices();
    return 0;
}
```