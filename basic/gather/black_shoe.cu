#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <stdio.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>

__device__ __host__ __inline__ float N(float x)
{
    return 0.5 + 0.5 * erf(x * M_SQRT1_2);
}

__device__ __host__ void price(float k, float s, float t, float r, float v, float *c, float *p)
{
    float srt = v * sqrtf(t);
    float d1 = (logf(s/k) + (r+0.5*v*v)*t) / srt;
    float d2 = d1 - srt;
    float kert = k * expf(-r * t);
    *c = N(d1)*s - N(d2)*kert;
    *p = kert - s + *c;
}

__global__ void price(float *k, float *s, float *t, float *r, float *v, float *c, float *p)
{
    int tid = threadIdx.x;
    price(k[tid], s[tid], t[tid], r[tid], v[tid], &c[tid], &p[tid]);
}

int main()
{
    const int count = 512;
    size_t size = count * sizeof(float);
    float *args[5];
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
    for (int i=0; i<5; ++i)
    {
        cudaMalloc(&args[i], size);
    }
    float *dc, *dp;
    //XXX: check cuda error
    cudaMalloc(&dc, size);
    cudaMalloc(&dp, size);
    //linear grid
    //It's a gather pattern, so the layout of the grid/block doesn't matter here.
    price<<<1, count>>> (args[0], args[1], args[2], args[3], args[4], dc, dp);
    cudaFree(dc);
    cudaFree(dp);
    float *hc = (float*)malloc(size);
    cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost);

    for (int i=0; i<10; ++i)
    {
        std::cout << hc[i] << std::endl;
    }
    free(hc);

}