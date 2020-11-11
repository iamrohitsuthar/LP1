#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <math.h>
#include <chrono>
#include <random>
#include <bits/stdc++.h>
using namespace std;

int random_in_range( int minimum, int maximum )
{
  thread_local std::ranlux48 rng( 
    std::chrono::system_clock::now().time_since_epoch().count() );
  return std::uniform_int_distribution <int> ( minimum, maximum )( rng );
}

__global__ void add(int *i, int *j, int *k) {
    int bid = blockIdx.x;
    k[bid] = i[bid] + j[bid];
}

void random_init(int *arr, int n) {
    for(int i = 0 ; i < n ; i++) {
        arr[i] = random_in_range(100,400);
    }
}

void add_cpu(int *i, int *j, int *k, int n) {
    for(int p = 0 ; p < n ; p++) {
        k[p] = i[p] + j[p]; 
    }
}

int main() {
    int n = 20000;
    int *a, *b;
    int c[n];
    int *i, *j, *k;
    int size = n * sizeof(int);

    a = new int[n];
    b = new int[n];
    random_init(a,n);
    random_init(b,n);

    cout<<"First: ";
    for(int i = 0 ; i < n ; i++) {
        cout<<a[i]<<" ";
    }
    cout<<endl;
    
    cout<<"Second: ";
    for(int i = 0 ; i < n ; i++) {
        cout<<b[i]<<" ";
    }
    cout<<endl;
    
    cudaMalloc((void **)&i,size);
    cudaMalloc((void **)&j,size);
    cudaMalloc((void **)&k,size);

    cudaMemcpy(i,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(j,b,size,cudaMemcpyHostToDevice);

    float gpu_elapsed_time;
    cudaEvent_t gpu_start,gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start,0);
    add<<<n,1>>>(i,j,k);
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);

    cudaMemcpy(c,k,size,cudaMemcpyDeviceToHost);

    cout<<endl;
    cout<<"GPU Elapsed time is: "<<gpu_elapsed_time<<" milliseconds";
    cout<<endl;
    cout<<"Parallel Result: ";
    for(int i = 0 ; i < n ; i++) {
        cout<<c[i]<<" ";
    }
    cout<<endl;
    cout<<endl;

    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start,0);
    add_cpu(a,b,c,n);
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);

    cout<<"CPU Elapsed time is: "<<gpu_elapsed_time<<" milliseconds";
    cout<<endl;

    cout<<"Serial Result: ";
    for(int i = 0 ; i < n ; i++) {
        cout<<c[i]<<" ";
    }
    cout<<endl;

    cudaFree(i);
    cudaFree(j);
    cudaFree(k);

    return 0;
}