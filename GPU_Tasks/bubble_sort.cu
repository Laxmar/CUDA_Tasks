#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "utility.h"
#include <cmath>

cudaError_t bubble_sort(float* array, unsigned int size);


__global__ void bubble_sort_kernel(float* dev_array, unsigned int s, unsigned int size)
{
    // s - even/odd stage indicator
    int i, j;
    float a, b;
    int index = 2 * (threadIdx.x + blockDim.x * blockIdx.x);
   /* if (index + s == 0) {
        printf("kernel 0 was run\n");
    }*/

    i = index + s; 
    j = i + 1;
    if (j < size) 
    { 
        a = dev_array[i];
        b = dev_array[j];
        if (b < a) 
        { 
            dev_array[i] = b;
            dev_array[j] = a;
        } 
    }
}

int main()
{
    const unsigned int array_size = 65536; // 262144;
    float a[array_size] = {};

    generate_array(a, array_size, -1000, 1000);
    //print_array(a, array_size);

    cudaError_t cudaStatus = bubble_sort(a, array_size);

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "bubble_sort failed!");
        return 1;
    }
    print_array(a, 10);
    //print_array(a, array_size);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    if (is_array_sorted(a, array_size))
    {
        printf("\n\nProgram finished without errros and Array is sorted\n");
    }
    else
    {
        printf("\n\nERROR: Array is not sorted!\n");
    }

    return 0;
}


cudaError_t bubble_sort(float* array, unsigned int array_size)
{
    float* dev_array = 0;
    int n = array_size / 2; // each thread use 2 numbers
    int threads_number = 512; // number of threads within each block
    int blocks = std::max(int(std::ceil(n / float(threads_number))), 1);

    printf("array_size: %d n: %d threads_number per block: %d blocks: %d\n\n", array_size, n, threads_number, blocks);

    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_array, array_size * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_array, array, array_size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); // TODO check status 
    cudaEventCreate(&stop); // TODO check status 
    cudaEventRecord(start, 0); // TODO check status 

    // launch Kernel
    for (unsigned int i = 0; i < array_size - 1; i++) 
    { 
        bubble_sort_kernel<<<blocks, threads_number>>> (dev_array, (i % 2), array_size);
       // kernel << <blocks, threads >> >
    }
    cudaEventRecord(stop, 0); // TODO check status 
    cudaEventSynchronize(stop); // TODO check status 
    float time;
    cudaEventElapsedTime(&time, start, stop); // TODO check status 
    std::cout << "time : " << time << "ms.\n";
    cudaEventDestroy(start); // TODO check status 
    cudaEventDestroy(stop); // TODO check status 

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(array, dev_array, array_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_array);

    return cudaStatus;
}

