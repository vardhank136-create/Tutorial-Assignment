
#include <stdio.h>
#include <cuda_runtime.h>

int main() {

    int deviceCount = 0;

    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    printf("Number of CUDA Devices: %d\n", deviceCount);

    for (int device = 0; device < deviceCount; device++) {

        cudaDeviceProp prop;

        cudaGetDeviceProperties(&prop, device);

        printf("\n============================================\n");
        printf("          CUDA DEVICE %d PROPERTIES\n", device);
        printf("============================================\n");

        printf("Device Name                 : %s\n", prop.name);

        printf("Compute Capability          : %d.%d\n",
               prop.major, prop.minor);

        printf("Total Global Memory         : %.2f GB\n",
               prop.totalGlobalMem /
               (1024.0 * 1024.0 * 1024.0));

        printf("Total Constant Memory       : %zu KB\n",
               prop.totalConstMem / 1024);

        printf("Shared Memory Per Block     : %zu KB\n",
               prop.sharedMemPerBlock / 1024);

        printf("Registers Per Block         : %d\n",
               prop.regsPerBlock);

        printf("Warp Size                   : %d\n",
               prop.warpSize);

        printf("Maximum Threads Per Block   : %d\n",
               prop.maxThreadsPerBlock);

        printf("Maximum Threads Dimension   : %d x %d x %d\n",
               prop.maxThreadsDim[0],
               prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);

        printf("Maximum Grid Size           : %d x %d x %d\n",
               prop.maxGridSize[0],
               prop.maxGridSize[1],
               prop.maxGridSize[2]);

        printf("Multiprocessors             : %d\n",
               prop.multiProcessorCount);

        printf("Clock Rate                  : %.2f GHz\n",
               prop.clockRate / 1000000.0);

        printf("Memory Clock Rate           : %.2f GHz\n",
               prop.memoryClockRate / 1000000.0);

        printf("Memory Bus Width            : %d bits\n",
               prop.memoryBusWidth);

        printf("L2 Cache Size               : %d KB\n",
               prop.l2CacheSize / 1024);

        printf("Concurrent Kernels          : %s\n",
               prop.concurrentKernels ? "Yes" : "No");

        printf("ECC Enabled                 : %s\n",
               prop.ECCEnabled ? "Yes" : "No");

        printf("Unified Addressing          : %s\n",
               prop.unifiedAddressing ? "Yes" : "No");

        printf("Managed Memory              : %s\n",
               prop.managedMemory ? "Yes" : "No");

        printf("Can Map Host Memory         : %s\n",
               prop.canMapHostMemory ? "Yes" : "No");

        printf("Async Engine Count          : %d\n",
               prop.asyncEngineCount);

        printf("Max Threads Per SM          : %d\n",
               prop.maxThreadsPerMultiProcessor);

        printf("Max Blocks Per SM           : %d\n",
               prop.maxBlocksPerMultiProcessor);

        printf("============================================\n");
    }

    return 0;
}
