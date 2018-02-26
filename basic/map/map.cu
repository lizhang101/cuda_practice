#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <iostream>
using namespace std;

__global__ void addTen(float *d, int count)
{
    //XXX: some of the parameters can be calculated on the host.
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    int thread_pos_in_block =   threadIdx.x + 
                                threadIdx.y * blockDim.x + 
                                threadIdx.z * blockDim.x * blockDim.y;
    int block_pos_in_grid = blockIdx.x + 
                            blockIdx.y * gridDim.x +
                            blockIdx.z * gridDim.x * gridDim.y;
    int tid = block_pos_in_grid * threads_per_block + thread_pos_in_block;
    int array[4];
    for (int i = 0; i < 4; i++)
    {
        //will be in local memory (?)
        array[i] = d[i];
    }
    //guard condition
    if (tid < count)
    {
        for (int i=0; i<4; i++)
        {
            d[tid] += array[i];
        }
    }
}

int main()
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(gen, time(0));

    const int count = 123456;
    const int size = count * sizeof(float);
    float *d;
    float h[count];
    cudaMalloc(&d, size);
    curandGenerateUniform(gen, d, count);
    dim3 block(8,8,8);
    dim3 grid(16, 16);
    addTen<<<grid, block>>> (d, count);
    cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);
    cudaFree(d);
    for (int i = 0; i < 10; ++i)
    {
        cout << h[i] << endl;
    }
}