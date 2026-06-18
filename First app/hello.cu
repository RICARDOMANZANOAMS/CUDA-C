#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello()
{
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main()
{
    printf("Starting program...\n");

    hello<<<1,8>>>();

    cudaError_t err = cudaGetLastError();
    printf("Launch error: %s\n", cudaGetErrorString(err));

    err = cudaDeviceSynchronize();
    printf("Sync error: %s\n", cudaGetErrorString(err));

    printf("Finished.\n");

    return 0;
}