all: program

program: mul_matrix.cpp
	g++ -o mul_matrix -L/usr/local/cuda/lib64 -lcuda -lcudart mul_matrix.cpp

clean: rm -rf *o program
