#include <iostream>
#include <cuda_runtime.h>

#define N 4
#define TILE 4

__global__ void matmultile(float *A, float *B, float *C)
{   
    //Shared memory defintion
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < N / TILE; t++)
    {   //Copy numbers from global memory to shared memory in small tiles
        As[threadIdx.y][threadIdx.x] =
            A[row * N + t * TILE + threadIdx.x];

        Bs[threadIdx.y][threadIdx.x] =
            B[(t * TILE + threadIdx.y) * N + col];

        __syncthreads();

        //Iterate through each tile multiplying tiles
        for (int k = 0; k < TILE; k++)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}

int main()
{
    float h_A[N * N] =
    {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,1,2,3,
        4,5,6,7
    };

    float h_B[N * N] =
    {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,1,2,3,
        4,5,6,7
    };

    float h_C[N * N];

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    matmultile<<<grid, block>>>(d_A, d_B, d_C);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Matrix C:\n";

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}