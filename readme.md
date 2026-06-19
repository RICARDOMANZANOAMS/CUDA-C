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

Understanding the difference between the CPU and GPU is the foundation of CUDA programming. A CUDA application uses the CPU to control the execution while offloading computationally intensive, highly parallel tasks to the GPU.
