all: program

program: black_shoe.cu
	nvcc -ptx -ptxas-options=-v -o black_shoe -L/usr/local/cuda/lib64 -lcuda -lcudart -lcurand black_shoe.cu

clean: rm -rf *o program
