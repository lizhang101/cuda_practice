#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <iostream>
using namespace std;

__global__ void runningSum(int *d)
{
    int tid = threadIdx.x;
    int threads = blockDim.x;
    //tc - total number of threads allowed.
    for (int tc = threads, step = 1; tc > 0; step <<= 1)
    {
        //guardian
        if (tid < tc)
        {
            d[tid+step] += d[tid];
        }
        tc -= step;
    }
}

int main()
{
    const int count = 32;
    const size_t size = count * sizeof(int);
    int h[count];
    for (int i=0; i<count; ++i)
    {
        h[i] = i+1;
    }

    int *d;
    cudaMalloc(&d, size);
    cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

    runningSum<<<1, count-1>>>(d);

    cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);
    cudaFree(d);
    for (int i=0; i < count; ++i)
    {
        std::cout << h[i] << std::endl;
    }
}