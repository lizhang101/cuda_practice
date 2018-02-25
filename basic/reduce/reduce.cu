#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <iostream>
using namespace std;

__global__ void sumSingleBlock(int *d)
{
    int tid = threadIdx.x;
    for (int tc = blockDim.x, stepSize = 1; tc > 0; tc >>=1, stepSize <<= 1)
    {
        if (tid < tc)
        {
            int pa = tid * stepSize * 2;
            int pb = pa + stepSize;
            d[pa] += d[pb];
        }
    }
}

__global__ void sumSingleBlock_shm(int *d)
{
    extern __shared__ int dcopy[];
    int tid = threadIdx.x;

    dcopy[tid*2] = d[tid*2];
    dcopy[tid*2+1] = d[tid*2+1];
    for (int tc = blockDim.x, stepSize = 1; tc > 0; tc >>=1, stepSize <<= 1)
    {
        if (tid < tc)
        {
            int pa = tid * stepSize * 2;
            int pb = pa + stepSize;
            dcopy[pa] += dcopy[pb];
        }
    }
    if (tid == 0) 
    {
        d[0] = dcopy[0];
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

    sumSingleBlock<<<1, count/2>>>(d);

    int result;
    cudaMemcpy(&result, d, sizeof(int), cudaMemcpyDeviceToHost);
    //cudaFree(d);
    std::cout << "use global mem:" << result << std::endl;

    cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);
    sumSingleBlock_shm<<<1, count/2, count>>>(d);

    cudaMemcpy(&result, d, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "use shared mem:" << result << std::endl;

    cudaFree(d);
}