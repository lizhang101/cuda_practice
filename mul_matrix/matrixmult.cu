#include <cstdlib>
#include <stdio.h>
#include <cassert>
#include <cuda_runtime.h>
/*
   Naive implementation.
   Allocate one thread for one element in result matrix, processing dot(Arow, Bcol);
*/
__global__ void kMatrixMul0 (float *d_res, 
                               float *d_mat1, int m1, int m2,
                               float *d_mat2, int n1, int n2){
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if (x >= n2 || y >= m1){return;}
    float sum=0.0;
    for (int i=0; i<m2; i++){
//    	if (blockIdx.x == 0 && threadIdx.x == 0 && blockIdx.y == 0 && threadIdx.y == 0) {
//    		printf("sum:%f mat1:%f mat2:%f i:%d\n", sum, d_mat1[y*m2+i], d_mat2[i*n2+x], i);
//    	}
        sum += d_mat1[y*m2+i] * d_mat2[i*n2+x];
    }
    d_res[n2*y+x] = sum;
};

/* Using shared memory 
*/
template<int BLOCK_SIZE>
__global__ void kMatrixMul1 (float *d_res, 
                               float *d_mat1, int m1, int m2,
                               float *d_mat2, int n1, int n2) {
    //assume squre block
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    //trivial opt: calculate these in CPU, shared by all blocks
    int aStart = blockDim.y*blockIdx.y * m2;
    int bStart = blockDim.x*blockIdx.x;
    int bStep = blockDim.y*n2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float c = 0.0f;
    for (int a = aStart, b = bStart; a < m2; a += blockDim.x, b += bStep) {
        //load data
        As[ty][tx] = d_mat1[a + m2 * ty + tx];   
        Bs[ty][tx] = d_mat2[b + n2 * ty + tx];
        __syncthreads();
        for (int k = 0; k<blockDim.x; k++){
            c += As[ty][k] * Bs[k][tx];
        }
    }
    d_res[(blockDim.y*blockIdx.y+ty)*n2+blockDim.x*blockIdx.x+tx] = c;
    //__syncthreads();
}


float* MatrixMultGPU0(float *mat1, int m1, int m2, float *mat2, int n1, int n2){
    float *d_res, *d_mat1, *d_mat2;
    cudaEvent_t start, end;
    cudaError_t error;
    error = cudaEventCreate(&start);
    error = cudaEventCreate(&end);



    //malloc the device memory for matrices 
    cudaError_t result = cudaMalloc((void**)&d_res, sizeof(float)*m1*n2);
    result = cudaMalloc((void**)&d_mat1, sizeof(float)*m1*m2);
    assert (result == cudaSuccess);
    result = cudaMalloc((void**)&d_mat2, sizeof(float)*n1*n2);
    assert (result == cudaSuccess);

    //init source matrices in device memory
    result = cudaMemcpy(d_mat1, mat1, sizeof(float)*m1*m2, cudaMemcpyHostToDevice);
    assert (result == cudaSuccess);
    result = cudaMemcpy(d_mat2, mat2, sizeof(float)*n1*n2, cudaMemcpyHostToDevice);
    assert (result == cudaSuccess);

    cudaEventRecord(start, NULL);

    int N = 32;

    dim3 block_size(N, N);
    //grid width in blocks
    int grid_wib = ceil(float(n2)/float(N));
    //grid height in blocks
    int grid_hib = ceil(float(m1)/float(N));
    dim3 grid_size(grid_wib, grid_hib);

    //naive version
    //kMatrixMul0<<<grid_size, block_size>>>(d_res, d_mat1, m1, m2, d_mat2, n1, n2);

    //sharedMem version
    kMatrixMul1<32><<<grid_size, block_size>>>(d_res, d_mat1, m1, m2, d_mat2, n1, n2);

    //copy back the multiplication result
    float* res = new float[m1*n2];
    result = cudaMemcpy(res, d_res, sizeof(float)*m1*n2, cudaMemcpyDeviceToHost);
    assert (result == cudaSuccess);

    cudaEventRecord(end, NULL);
    error = cudaEventSynchronize(end);

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, end);
    printf("calculation Time:%f ms\n", msecTotal);

    cudaFree(d_res);
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    return res;
}

float* MatrixMulCPU(float *mat1, int m1, int m2, float *mat2, int n1, int n2){
    assert(m2 == n1 && "matrx a's cols != matrix b's rows");
    
    float* res = new float[m1*n2] ;
    for (int i = 0; i < m1; i++) {
        for (int j = 0; j < n2; j++) {
            res[i*m1 + j] = 0;
            for (int x = 0; x < m2; x++) {
                res[i*m1+j] += mat1[i*m1+x] * mat2[x*n1+j];
            }
        }
    }
    return res;

}

float CompareMatrix(float mat1[], float mat2[], int m1, int m2){
    float err = 0;
    for (int x=0; x<m1; x++){
        for (int y=0; y<m2; y++){
            err += mat1[m2*y+x] - mat2[m2*y+x];
        }
    }
    return err;
    /*
    if (err > 0.1){
        printf("matrix comparison failed.error:%f\n", err); 
        return false;
    }
    return true;
    */
}

void FillMatrix(float mat[], int m1, int m2, float d=1.0){
    for (int i=0; i<m1; i++){
        for (int j=0; j<m2; j++){
            if (d<0.0) {
                mat[i*m1+j] = static_cast<float>(rand())/static_cast<float>(RAND_MAX) * 10.0;
            } else {
                mat[i*m1+j] = d;
            }
        }
    }
}
void PrintMatrix(float mat[], int m1, int m2){
    for (int i=0; i<m1; i++){
        for (int j=0; j<m2; j++){
            printf("%f,", mat[i*m1+j]);
        }
        printf("\n");
    }
}


int main(int argc, char *argv[]){
    int m1 = 1024;
    int m2;
    int n1 = m2 = 1024;
    int n2 = 1024;
    float *matrix_a = new float[m1*m2];
    float *matrix_b = new float[n1*n2];
    FillMatrix((float*)matrix_a, m1, m2);
    FillMatrix((float*)matrix_b, n1, n2);
    //float *ref = MatrixMulCPU((float*)matrix_a, m1, m2, (float*)matrix_b, n1, n2);
    float *res = MatrixMultGPU0((float*)matrix_a, m1, m2, (float*)matrix_b, n1, n2);
    //printf("error:%f\n", CompareMatrix(ref, res, m1, n2));
    //PrintMatrix(res, m1, m2);
    delete[] matrix_a;
    delete[] matrix_b;
    delete res;
    //delete ref;
}
