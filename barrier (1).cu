
#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_THREADS 8

__device__ volatile int counter = 0;

__device__ void barrier()
{
    __syncthreads();

    atomicAdd((int*)&counter, 1);

    while (counter < NUM_THREADS)
    {
    }

    __syncthreads();
}

__global__ void testBarrier()
{
    int tid = threadIdx.x;

    printf("Thread %d reached barrier\n", tid);

    barrier();

    printf("Thread %d passed barrier\n", tid);
}

int main()
{
    printf("Starting CUDA Barrier Program\n\n");

    int zero = 0;

    cudaMemcpyToSymbol(
        counter,
        &zero,
        sizeof(int)
    );

    testBarrier<<<1, NUM_THREADS>>>();

    cudaError_t error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
    {
        printf("CUDA Error: %s\n",
               cudaGetErrorString(error));

        return 1;
    }

    printf("\nAll threads completed successfully!\n");

    return 0;
}
