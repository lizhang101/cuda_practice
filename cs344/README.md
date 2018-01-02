cs344
=====

Solutions to Udacity's parallel computing class problem sets.

# How to Build

**All the solutions are in student_func.cu in each HW directory. **

```
mkdir build
cd build
cmake ..
make
```

# Problems Sets

## HW1 : RGB TO Gray
Just to show some basic Cuda usage.

## HW2: Gaussian Blur

###implementation:
#### Naive implementation: 
Not using shared memory. One thread working on 9x9 input and output 1 filtered result. Use global memory loads for filter weights and image inputs.

#### Optimization:
1. use constant memory for filter weight. Noticeable improvement. 

2. Loop unrolling for the filtering process since the operation in the loop is very simple. Noticeable improvement. 

3. Using Shared memory. 
  image_tile[TileH+FilterW] \[TiileW+FilterW] to load the input data for this block, saying size BlockH x BlockW.
  some threads will have to load extra data. Use multiple blocks to cover the whole input data region. Disable loading for the threads outside the input data region. Repeat at the edge of the input picture. Note there would be divergense in some warps, and workloads per threads are not balanced. The threads of the upper-left corner in the block have the most workloads to do. **It would be better to distribute the loading evenly, and use adjacent threads to load the extra regions for better memory efficency, but I didn't do this in the practice**
  Initially used 32x32 for block size.

4. Use wide rectangle for block.
  Clearly, 32x32 block size is not good for this case. It's better to use wider rectangles. ideally, (k*32 + 8) % 32 ==0, which means the minimal ideal width is 256. Thus the address is algned to 32B and would have divergnese. But it's too big for 9x256 threads to fit in one block. 

5. processing RGB in one kernel instead of using 3 kernels. (TODO)

   The original approach in this practice code separated the RGB channels into 3 images. It should be able to use vector and process RGB in one kernel.

6. one thread loads more data (TODO)
  ​

7. Register Blocking (TODO)

8. Software prefetch (TODO)

9. Optimize for SMEM bank conflicts and Register Bank Conflict. 

   Used nvvp, SMEM has 0 bank conflicts.




