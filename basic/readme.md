# Introduction

## Programming Patterns

### map

Apply same operations on each thread.

### Gather

Each thread reads from several places.

1. used __device__ qualifier, A function called from GPU and runs on GPU.
2. A function can be compiled for both GPU and CPU.( __device__ and __host__)

## Scatter

Each thread writes to several locations.
Beware of thread collisions.

### Reduce

Operation should be associative.
For example, sum.

* Adding up N data elements.
* use 1 block of N/2 threads.
* Each threads does: x[i] += x[j];
* At each iteration:
  * #of threads halved
  * step size(j-i) doubled
* x[0] is the result.

### Scan

Accumulation.
Each output value yn is calculated as a function involving inputs from 1 to n.
E.g. A running sum of elements y[n] = sum<i=[1:n]>(x[i])

* Similar to reduce.Work over the list, but balanced to the right.(Reduce is balanced to the left).
* Require N-1 threads.
* Step size keeps doubling
* number of threads reduced by step size.
* each thread n does: x[n+step] += x[n];

## Memory

### Device Memory

* Grid scope
* application lifetime
* Dynamic
  * cudaMalloc()
  * Pass pointers to kernel (float *a, cudaMalloc(&a, size))
  * cudaMemcpy()
  * cudaFree()
* Static
  * Declare global variable as device  
    __device__ int sum = 0;
  * Use freely within the kernel
  * use cudaMemcpy[To/From]Symbol() to copy to/from host memory
  * No need to explicitly deallocate.
* slowest

### Constant & Texture Memory

* Read-only. useful for lookup tables, model parameters, etc.
* Grid scope, appolication lifetime.
* Resides in device memory, but 
* Cached in a constant memory cache.
* constrained by MAX_CONSTANT_MEMORY
  * Expect 64kB
* Similar operation to statically-defined device memory.
  * Declare as __constant__
  * Use freely within the kernel
  * use CudaMemcpy[TO/From]Symbol()
* very fast provided all threads read from the same location
* used for kernel arguments
* Texture Memory: similar to Constant, optimized for 2D access patterns.

### shared memory

* Block scope
  * shared only within a thread block
  * not shared between blocks
* kernel lifetime.
* must be declared within the kernel function body.
* Very fast.

### Register and local memory

* Memory can be allocated right within the kernel
  * Thread scope, kenel lifetime
* non-array memory
  * int tid = ...
  * stored in a register
  * very fast
* Array memory
  * Stored in "local memory"
  * local memory is an abstraction, actually put in global memory
  * slow as global memory

## Thread Cooperation and Synchronization

* barrier: __syncthreads()
* within a block

## Atomic

* Grid scope.
* atomic operations ensure that only one thread can access the location
* atomicOP(x, y)
  * t1 = *x; t2 = t1 OP y; *x = t2;
* #include "sm_20_atomic_functions.h"
