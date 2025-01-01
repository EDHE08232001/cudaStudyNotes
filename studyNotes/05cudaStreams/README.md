# CUDA Streams and Events

So far in the CUDA studying, we follow the bemow model of writing CUDA programs

**Kernel Level Parallelism**
```
Data Transfer from Host to Device  -->  Kernel Execution  -->  Data Transfer from Device to Host
------------------------------------------------------------------------------------------------> (Time)
```

---

## Grid Level Concurrency/Parallelism

Concurrency achieved by launching multiple kernels to the same device simultaneously and overlapping memory transfers with kernel execution.

![Grid Level Concurrency Diagram](./images/gridConcurrencyDiagram.webp)