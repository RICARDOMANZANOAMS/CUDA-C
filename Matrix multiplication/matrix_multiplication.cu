#include <stdio.h>
#include <cuda_runtime.h>

//------------------------------------------------------------
// CUDA Kernel
// Each thread computes ONE element of the output matrix C.
//------------------------------------------------------------
__global__ void matmul(float *A, float *B, float *C, int N)
{
    // Compute the row and column that this thread is responsible for.
    // blockIdx   -> identifies the current block.
    // blockDim   -> number of threads per block.
    // threadIdx  -> identifies the current thread inside the block.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    /*
    Example:

    Suppose we launch:

        dim3 blockSize(2,2);
        dim3 gridSize(1,1);

    This means:

        blockDim.x = 2
        blockDim.y = 2

        blockIdx.x = 0
        blockIdx.y = 0

    because there is only ONE block.
     */
    // Make sure the thread does not access memory outside the matrix.
    if (row < N && col < N)
    {
        float sum = 0.0f;

        // Compute the dot product between
        // row 'row' of matrix A and
        // column 'col' of matrix B.
        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        // Store the computed value into matrix C.
        C[row * N + col] = sum;
    }
}

int main()
{
    //------------------------------------------------------------
    // Matrix size
    //------------------------------------------------------------
    const int N = 2;

    //------------------------------------------------------------
    // Host (CPU) matrices stored in row-major order
    //
    // A = |1 2|
    //     |3 4|
    //
    // B = |5 6|
    //     |7 8|
    //------------------------------------------------------------
    float h_A[N * N] = {1, 2,
                        3, 4};

    float h_B[N * N] = {5, 6,
                        7, 8};

    float h_C[N * N];

    //------------------------------------------------------------
    // Device pointers
    //------------------------------------------------------------
    float *d_A;
    float *d_B;
    float *d_C;

    //------------------------------------------------------------
    // Allocate memory on the GPU
    //------------------------------------------------------------
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    //------------------------------------------------------------
    // Copy matrices from CPU (Host) to GPU (Device)
    //------------------------------------------------------------
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    /*
    ------------------------------------------------------------
     Configure the execution
    
     block = (2 x 2) threads
    
    Thread layout
    
     (0,0) (1,0)
     (0,1) (1,1)
    ------------------------------------------------------------
    */
    dim3 block(2, 2);

    //------------------------------------------------------------
    // Only one block is needed because the matrix is only 2x2.
    //------------------------------------------------------------
    dim3 grid(1, 1);

    //------------------------------------------------------------
    // Launch the kernel
    //------------------------------------------------------------
    matmul<<<grid, block>>>(d_A, d_B, d_C, N);

    //------------------------------------------------------------
    // Wait until all GPU threads finish.
    //------------------------------------------------------------
    cudaDeviceSynchronize();

    //------------------------------------------------------------
    // Copy the result back to the CPU.
    //------------------------------------------------------------
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    //------------------------------------------------------------
    // Print the resulting matrix.
    //------------------------------------------------------------
    printf("Matrix C:\n");

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%6.1f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    //------------------------------------------------------------
    // Free GPU memory.
    //------------------------------------------------------------
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}