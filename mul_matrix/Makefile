all: program

program: mul_matrix.cpp
	g++ -o matrixmult -L/usr/local/cuda/lib64 -lcuda -lcudart matrixmult.cu

clean: rm -rf *o program
