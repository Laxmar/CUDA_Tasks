#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "utility.h"


cudaError_t bubble_sort(float* array, unsigned int size);


__global__ void bubble_sort_kernel(float* dev_array, unsigned int s, unsigned int size)
{
    // s - even/odd stage indicator
    int i, j;
    float a, b;
    int index = 2 * (threadIdx.x + blockDim.x * blockIdx.x);

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
    const int array_size = 6;
    float a[array_size] = {};

    generate_array(a, array_size, -10, 10);
    //print_array(a, array_size);

    cudaError_t cudaStatus = bubble_sort(a, array_size);

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "bubble_sort failed!");
        return 1;
    }
    print_array(a, array_size);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    printf("Program finished without errros");

    return 0;
}


cudaError_t bubble_sort(float* array, unsigned int array_size)
{
    float* dev_array = 0;
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

    // launch Kernel
    for (unsigned int i = 0; i < array_size - 1; i++) 
    { 
        bubble_sort_kernel<<<1, array_size>>> (dev_array, (i % 2), array_size);
       // kernel << <blocks, threads >> >
    }

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