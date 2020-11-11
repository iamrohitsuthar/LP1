#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <chrono>
#include <random>
using namespace std;


int random_in_range( int minimum, int maximum )
{
  thread_local std::ranlux48 rng( 
    std::chrono::system_clock::now().time_since_epoch().count() );
  return std::uniform_int_distribution <int> ( minimum, maximum )( rng );
}

__global__
void matrixMultiplication(int *a, int *b, int *c, int m, int n, int k)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int sum=0;
   
    if(col<k && row<m) {
      for(int j=0;j<n;j++)
      {
          sum += a[row*n+j] * b[j*k+col];
      }
      c[k*row+col]=sum;
    }
    
}

void matrix_multiplication_cpu(int *a, int *b, int *c, int m, int n, int k) {
    for(int i = 0 ; i < m ; i++) {
        for(int j = 0 ; j < n ; j++) {
            long result = 0;
            for(int p = 0 ; p < k ; p++) {
                result=result+a[i*k+p]*b[p*k+j]; 
            }
            c[k*i+j] = result;
        }
    }
}

void init_result(int *a, int m, int k) {
    for(int i=0; i<m; i++) {
      for(int j=0; j<k; j++) {
        a[i*k + j] = 0;
      }
    }
}

void init_matrix(int *a, int n, int m) {
    for(int i=0; i<n; i++) {
      for(int j=0; j<m; j++) {
        a[i*m + j] = random_in_range(10,30);
      }
    }
}

void print_matrix(int *a, int n, int m) {
    for(int i=0; i<n; i++) {
      for(int j=0; j<m; j++) {
        cout<<"  "<<a[i*m + j];
      }
      cout<<endl;
    }
    cout<<endl;
}

int main()
{
    
    int *a,*b,*c;
    int *a_dev,*b_dev,*c_dev;
    int m=30, n=30, k=30;
    
    a = new int[m*n];
    b = new int[n*k];
    c = new int[m*k];
    
    init_matrix(a, m, n);
    init_matrix(b, n ,k);
    init_result(c, m, k);
    
    cout<<"First matrix : "<<endl;
    print_matrix(a, m, n);
    cout<<"Second matrix : "<<endl;
    print_matrix(b, n, k);
    
    cudaMalloc(&a_dev, sizeof(int)*m*n);
    cudaMalloc(&b_dev, sizeof(int)*n*k);
    cudaMalloc(&c_dev, sizeof(int)*m*k);
       
    cudaMemcpy(a_dev, a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b, sizeof(int)*n*k, cudaMemcpyHostToDevice);
    
    dim3 dimGrid(1,1);
    dim3 dimBlock(n,n);
    
    float gpu_elapsed_time;
    cudaEvent_t gpu_start,gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start,0);
    matrixMultiplication<<<dimGrid, dimBlock>>>(a_dev,b_dev,c_dev, m, n, k);
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    cout<<"GPU Elapsed time is: "<<gpu_elapsed_time<<" milliseconds"<<endl;
    
    cudaMemcpy(c, c_dev, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
    
    cout<<"GPU Result : "<<endl;
    print_matrix(c, m, k);
    cout<<endl;

    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start,0);
    matrix_multiplication_cpu(a, b, c, m, n, k);
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    cout<<"CPU Elapsed time is: "<<gpu_elapsed_time<<" milliseconds"<<endl;
 
    cout<<"CPU Result : "<<endl;
    print_matrix(c, m, k);
    
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
    
    delete[] a;
    delete[] b;
    delete[] c;
    
    return 0;
}