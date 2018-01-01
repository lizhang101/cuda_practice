cs344
=====

Solutions to Udacity's parallel computing class problem sets.

```
mkdir build
cd build
cmake ..
make
```

#HW1
rgb to gray.

## Naive implementation:

Not using shared memory. One thread working on 9x9 input and output 1 filtered result. Use global memory loads for filter weights and image inputs.

## Opt1: 

use constant memory for filter weight. Noticeable improvement. 

## Opt2:

Loop unrolling for the filtering process since the operation in the loop is very simple. Noticeable improvement. 

## Opt3: 

Using Shared memory. 

image_tile[TileW+FilterW] \[TiileW+FilterW]

some threads will load extra data.

There are several mapping methods:

1. each warp loads only 32 pixels, and output 24. Of cause the warp is not fully utilized, but the loading will be simple.
2. each warp loads 32+FilterW pixels, and output 32. Need to consider loading:
   1. definition: ix: image space x, tx: threadIdx.x
   2. first load 32 pixels [ix - 4, +32], then [ix-4+32, 32], disable read for ix>36. Will have some waste here.

## Opt4:

Clearly, 32x32 block size is not good for this case. It's better to use wider rectangles. ideally, (k*32 + 8) % 32 ==0, which means the minimal ideal width is 256. But it's too big for 9x256 threads in one block.

## Opt5:

processing RGB in one kernel instead of with 3 kernels.

## Opt5:

Register Blocking

## Opt6:

Optimize for SMEM bank conflicts and Register Bank Conflict. 




