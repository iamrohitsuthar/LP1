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
void matrixVector(int *vec, int *mat, int *result, int n, int m)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int sum=0;
    
    if(tid <= n) {
        for(int i=0; i<n; i++) {
            sum += vec[i]*mat[(i*m) + tid];
        }
        result[tid] = sum;
    }
}

void maxtrixVector_cpu(int *vec, int *mat, int *result, int n, int m) {
    for(int i = 0 ; i < n ; i++) {
        long sum = 0;
        for(int j = 0 ; j < m ; j++) {
            sum = sum + mat[j*m+i] * vec[j];  
        }
        result[i] = sum;
    }
}

void init_array(int *a, int n) {
    for(int i=0; i<n; i++)
      a[i] = random_in_range(10,40);
}

void init_matrix(int *a, int n, int m) {
    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            a[i*m + j] = random_in_range(10, 40);        
        }
    }
}

void print_array(int *a, int n) {
    for(int i=0; i<n; i++) {
        cout<<a[i]<<" ";
    }
    cout<<endl;
}

void print_matrix(int *a, int n, int m) {
    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++)
          cout<<"  "<<a[i*m + j];
        cout<<endl;
    }
}

int main() {
    int *a, *b, *c;
    int *a_dev, *b_dev, *c_dev;
    
    int n = 100;
    int m = 100;
    
    a = new int[n];
    b = new int[n*m];
    c = new int[m];
    
    init_array(a, n);
    init_matrix(b, n, m);
        
    cout<<"Initial vector array : "<<endl;
    print_array(a, n);
    cout<<endl;
    cout<<"Initial matrix : "<<endl;
    print_matrix(b, n, m);
    cout<<endl;
    
    cudaMalloc(&a_dev, sizeof(int)*n);
    cudaMalloc(&b_dev, sizeof(int)*n*m);
    cudaMalloc(&c_dev, sizeof(int)*m);
    
    cudaMemcpy(a_dev, a, sizeof(int)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b, sizeof(int)*n*m, cudaMemcpyHostToDevice);
    
    float gpu_elapsed_time;
    cudaEvent_t gpu_start,gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start,0);
    matrixVector<<<m, 1>>>(a_dev, b_dev, c_dev, n, m);
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    cout<<"GPU Elapsed time is: "<<gpu_elapsed_time<<" milliseconds"<<endl;

    cudaMemcpy(c, c_dev, sizeof(int)*m, cudaMemcpyDeviceToHost);
    
    cout<<"GPU Resultant vector : ";
    print_array(c, m);
    cout<<endl;

    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start,0);
    maxtrixVector_cpu(a, b, c, n, m);
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    cout<<"CPU Elapsed time is: "<<gpu_elapsed_time<<" milliseconds"<<endl;
    
    cout<<"CPU Resultant vector : ";
    for(int i = 0 ; i < n ; i++) {
        cout<<c[i]<<" ";
    }
    cout<<endl;
    
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
    
    delete[] a;
    delete[] b;
    delete[] c;
    
    return 0;
}
