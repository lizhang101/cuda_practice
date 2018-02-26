all: program

program: map.cu
	nvcc -keep -o map -L/usr/local/cuda/lib64 -lcuda -lcudart -lcurand map.cu

clean: rm -rf *o program
