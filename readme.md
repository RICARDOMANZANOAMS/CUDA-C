# Important Concepts

In this project, we will cover CUDA programming from the basics to advanced concepts using the C programming language. The goal is to understand how to execute computations directly on the GPU to accelerate parallel workloads.

## Host and Device
![GPU Architecture](Images/CPU-GPU.png)

CUDA programs execute on two different processors:

* **Host:** The CPU, responsible for running the main application, managing memory, and launching GPU kernels.
* **Device:** The GPU, responsible for executing thousands of threads in parallel.

The CPU prepares the data and transfers it to the GPU. Once the computation is complete, the results are copied back from the GPU to the CPU.

```
CPU (Host)
    |
    |  Transfer data
    v
GPU (Device)
    |
    |  Execute parallel computation
    v
Results returned to the CPU
```

## CPU vs GPU

The CPU is designed for low-latency execution and is optimized for sequential tasks, complex logic, and operating system management.

The GPU is designed for high-throughput computing. It contains thousands of lightweight cores that can execute many threads simultaneously, making it ideal for data-parallel algorithms where the same operation is performed on many data elements.

### Typical Use Cases

**CPU**

* Complex decision-making
* Sequential algorithms
* Input/Output operations
* Operating system tasks

**GPU**

* Matrix operations
* Image and video processing
* Scientific computing
* Machine learning and deep learning
* Physics simulations
* Large-scale parallel computations

## GPU Execution Model
![GPU Execution Model](Images/Grid-block-threads.png)

CUDA organizes the execution of parallel programs into a hierarchy of threads, blocks, and grids.

### Thread

A **thread** is the smallest execution unit on the GPU. Each thread executes the same kernel (function) but operates on different data. Every thread has a unique identifier (`threadIdx`) that allows it to determine which portion of the data it should process.

### Block

A **block** is a group of threads that execute on the same Streaming Multiprocessor (SM). Threads within the same block can:

* Communicate through **shared memory**.
* Synchronize their execution using `__syncthreads()`.

Blocks are independent of one another, allowing the GPU to schedule them efficiently across multiple SMs.

### Grid

A **grid** is a collection of one or more blocks. When a CUDA kernel is launched, the programmer specifies the number of blocks in the grid and the number of threads in each block.

```
Grid
 ├── Block 0
 │    ├── Thread 0
 │    ├── Thread 1
 │    └── ...
 ├── Block 1
 │    ├── Thread 0
 │    ├── Thread 1
 │    └── ...
 └── ...
```

### Warp

A **warp** is a group of **32 threads** within a block that execute instructions together on the GPU. The GPU schedules and executes threads at the warp level, not individually.

For maximum performance, threads within a warp should follow the same execution path. If different threads in the same warp take different branches of an `if` statement, the warp experiences **branch divergence**, which can reduce performance.

