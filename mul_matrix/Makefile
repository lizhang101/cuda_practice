all: program

program: matrixmult.cu
	nvcc -o matrixmult -L/usr/local/cuda/lib64 -lcuda -lcudart matrixmult.cu -g

clean: rm -rf *o program
