
#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_THREADS 8

// Global counter in GPU memory
__device__ int counter = 0;


// Barrier function
__device__ void barrier()
{
    // Make sure all threads reach this point
    __syncthreads();

    // Only thread 0 updates the counter
    if (threadIdx.x == 0)
    {
        atomicAdd(&counter, 1);
    }

    // Wait until thread 0 has updated the counter
    __syncthreads();

    // Thread 0 waits until all threads have reached
    // the barrier
    if (threadIdx.x == 0)
    {
        while (counter < 1)
        {
            // wait
        }
    }

    // Release all threads
    __syncthreads();
}


// CUDA kernel
__global__ void testBarrier()
{
    int tid = threadIdx.x;

    printf("Thread %d reached barrier\n", tid);

    // Call barrier
    barrier();

    printf("Thread %d passed barrier\n", tid);
}


int main()
{
    printf("Starting CUDA Barrier Program\n\n");

    // Reset counter
    int zero = 0;

    cudaMemcpyToSymbol(
        counter,
        &zero,
        sizeof(int)
    );

    // Launch kernel
    testBarrier<<<1, NUM_THREADS>>>();

    // Wait for kernel to finish
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
