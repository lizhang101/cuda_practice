#include <stdio.h>
__global__ void matrix_add(float *c, float *a, float *b) {
	int x = threadIdx.x;
	c[x] = a[x] + b[x];
}

void matrix_add_cpu(float *c, float *a, float *b, int size) {
	for (int i=0; i<size; i++){
		c[i] = a[i] + b[i];
	}

}

void print_matrix(float *m, int size){
    for (int i=0; i<size; i++){
        printf("m[%d]=%f\n", i, m[i]);
	}
}

bool compare (float *ref, float *gpu_rlt, int size){
    for (int i = 0; i<size; i++) {
        if (ref[i] != gpu_rlt[i]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv){
	const int ARRAY_SIZE = 96;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	//generate matrix A and B
	float A[ARRAY_SIZE];
	float B[ARRAY_SIZE];
	for (int i=0; i<ARRAY_SIZE; i++){
		A[i] = float(i);
		B[i] = float(i);
	}
	//for results
	float C[ARRAY_SIZE];
	float ref_results[ARRAY_SIZE];

	// alloc memory on gpu
	float *d_c, *d_a, *d_b;
	cudaMalloc((void**) &d_c, ARRAY_BYTES);
	cudaMalloc((void**) &d_a, ARRAY_BYTES);
	cudaMalloc((void**) &d_b, ARRAY_BYTES);

	// copy A and B to gpu
	cudaMemcpy(d_a, A, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// invoke kernel
	matrix_add<<<1, ARRAY_SIZE>>>(d_c, d_a, d_b);

	cudaMemcpy(C, d_c, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	print_matrix(C, ARRAY_SIZE);

    matrix_add_cpu(ref_results, A, B, ARRAY_SIZE);
    if (compare(ref_results, C, ARRAY_SIZE)) {
        printf("Correct\n");
    } else {
        printf ("Wrong\n");
    }
}
