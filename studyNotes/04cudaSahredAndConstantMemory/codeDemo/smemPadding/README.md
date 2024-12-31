# **Enhancements and Additions**
1. **Documentation and Comments**:
   - Added detailed comments for each kernel to explain the purpose of padding and memory access patterns.
   - Described index calculations for row-major and column-major formats.

2. **Host Code**:
   - Integrated the kernel launches into `main`.
   - Added printing logic for visualizing the matrix layout.

3. **Static vs. Dynamic Memory**:
   - Demonstrated the use of both statically and dynamically allocated shared memory.

4. **Padding**:
   - Explained how padding (`IPAD`) helps avoid **bank conflicts** by disrupting alignment issues in shared memory access.

---

### **Expected Output**

For a `32x32` matrix with padding, the output matrices should demonstrate the effects of row-major writes and column-major reads, showing a transformation of memory layout.

#### Output: `setRowReadColPad`
```
=== Kernel: setRowReadColPad ===
   0   32   64   96  ...
   1   33   65   97  ...
   2   34   66   98  ...
   ...
```

#### Output: `setRowReadColDynamicPad`
```
=== Kernel: setRowReadColDynamicPad ===
   0   32   64   96  ...
   1   33   65   97  ...
   2   34   66   98  ...
   ...
```

---

### **Key Learnings**
- **Padding** is a simple yet effective method to prevent **bank conflicts**.
- Proper synchronization (`__syncthreads()`) is crucial to avoid race conditions in shared memory.
- Understanding **memory access patterns** is vital for optimizing CUDA kernel performance.

This implementation demonstrates the versatility of shared memory in CUDA and highlights the importance of efficient memory access in GPU programming.