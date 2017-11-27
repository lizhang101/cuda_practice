#include <cstdlib>
#include <stdio.h>
#include <cassert>

int* MatrixMulCPU(int *mat1, int m1, int m2, int *mat2, int n1, int n2){
    assert(m2 == n1 && "matrx a's cols != matrix b's rows");
    
    int* res = new int[m1*n2] ;
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

void FillMatrix(int mat[], int m1, int m2){
    for (int i=0; i<m1; i++){
        for (int j=0; j<m2; j++){
            mat[i*m1+j] = rand()%1000;
        }
    }
}
void PrintMatrix(int mat[], int m1, int m2){
    for (int i=0; i<m1; i++){
        for (int j=0; j<m2; j++){
            printf("%d,", mat[i*m1+j]);
        }
        printf("\n");
    }
}
int main(int argc, char *argv[]){
    int m1 = 10;
    int m2;
    int n1 = m2 = 10;
    int n2 = 10;
    int matrix_a[m1][m2];
    int matrix_b[n1][n2];
    FillMatrix((int*)matrix_a, m1, m2);
    FillMatrix((int*)matrix_b, n1, n2);
    int *res = MatrixMulCPU((int*)matrix_a, m1, m2, (int*)matrix_b, n1, n2);
    PrintMatrix(res, m1, m2);
    delete res;

}
