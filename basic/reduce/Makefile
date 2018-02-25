all: program

program: reduce.cu
	nvcc -o reduce -L/usr/local/cuda/lib64 -lcuda -lcudart -lcurand reduce.cu

clean: rm -rf *o program
