# CUDA Shared Memory: Row-Major vs Column-Major Access Patterns

This project demonstrates the concepts of shared memory in CUDA, focusing on **row-major** and **column-major** access patterns. The code includes CUDA kernels that explore how shared memory accesses affect performance and behavior depending on the memory layout and access modes.

---

## **Key Concepts**

### **Shared Memory**
Shared memory is an on-chip memory accessible by all threads within a thread block. It is significantly faster than global memory but limited in size. Efficient utilization of shared memory is crucial for optimizing CUDA kernel performance.

### **Row-Major vs Column-Major Access**
- **Row-Major Access**: Memory is accessed sequentially row by row.
- **Column-Major Access**: Memory is accessed sequentially column by column.
- Access patterns significantly affect the performance due to **shared memory bank conflicts**.

---

## **Code Overview**

This repository contains three CUDA kernels that showcase different access patterns to shared memory:

### **Kernels**

#### 1. `setRowReadCol`
- **Writes in Row-Major Order**: Each thread writes to the shared memory in row-major format.
- **Reads in Column-Major Order**: Each thread reads from the shared memory in column-major format.
- **Outcome**: Causes **bank conflicts** during reads, as all threads in a warp access elements from the same memory bank.

#### 2. `setColReadRow`
- **Writes in Column-Major Order**: Each thread writes to the shared memory in column-major format.
- **Reads in Row-Major Order**: Each thread reads from the shared memory in row-major format.
- **Outcome**: Causes **bank conflicts** during writes, as threads in a warp write to the same memory bank.

#### 3. `setRowReadRow`
- **Writes in Row-Major Order**: Each thread writes to the shared memory in row-major format.
- **Reads in Row-Major Order**: Each thread reads from the shared memory in row-major format.
- **Outcome**: No bank conflicts, as threads in a warp access different memory banks.

---

## **Memory Layout**

The shared memory is divided into **banks** (typically 32 banks for modern GPUs). Each thread in a warp ideally accesses a different bank for maximum parallelism. If multiple threads in a warp access the same bank, **bank conflicts** occur, and memory transactions are serialized.

### Example: 32x32 Matrix in Shared Memory (Row-Major)
```
|  B1  |  B2  |  B3  |  B4  | .... |  B32  |
   0      1      2      3            31
   32     33     34     35           63
   64     65     66     67           95
   ...    ...    ...    ...          ...
```

### Row-Major Access Pattern
- Threads in a warp access different banks.
- **No bank conflicts**.

### Column-Major Access Pattern
- Threads in a warp access the same bank (e.g., threads access elements `0`, `32`, `64`, ... from bank `0`).
- **Causes bank conflicts**.

---

## **Code Execution**

### **Requirements**
- NVIDIA GPU with CUDA support.
- CUDA Toolkit installed.

### **Compile and Run**
1. Compile the code:
   ```bash
   nvcc shared_memory_demo.cu -o shared_memory_demo
   ```
2. Run the executable:
   ```bash
   ./shared_memory_demo
   ```

### **Output Example**
The program displays the matrix results for each kernel. For example:

#### **Output: setRowReadCol**
The output shows the effects of row-major writes and column-major reads:
```
=== Kernel: setRowReadCol ===
   0    32    64    96   ...
   1    33    65    97   ...
   2    34    66    98   ...
   ...
```

#### **Output: setColReadRow**
The output shows the effects of column-major writes and row-major reads:
```
=== Kernel: setColReadRow ===
   0     1     2     3   ...
  32    33    34    35   ...
  64    65    66    67   ...
  ...
```

#### **Output: setRowReadRow**
The output shows row-major writes and reads (ideal case with no bank conflicts):
```
=== Kernel: setRowReadRow ===
   0     1     2     3   ...
  32    33    34    35   ...
  64    65    66    67   ...
  ...
```

---

## **Key Takeaways**

1. **Bank Conflicts**:
   - Row-major access avoids conflicts as threads access different banks.
   - Column-major access causes conflicts when threads access the same bank.

2. **Performance Optimization**:
   - Align access patterns with the shared memory layout to maximize bandwidth.
   - Use **padding** if column-major access is unavoidable to disrupt bank conflicts.

3. **Shared Memory Utilization**:
   - Understand the bank structure and memory access patterns for optimal kernel performance.