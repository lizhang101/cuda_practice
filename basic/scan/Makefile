all: program

program: scan.cu
	nvcc -o scan -L/usr/local/cuda/lib64 -lcuda -lcudart -lcurand scan.cu

clean: rm -rf *o program
