#include <iostream>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <random>
#include <string.h>
#include <stdlib.h>

using namespace std;
using namespace std::chrono;

int random_in_range( int minimum, int maximum )
{
  thread_local std::ranlux48 rng( 
    std::chrono::system_clock::now().time_since_epoch().count() );
  return std::uniform_int_distribution <int> ( minimum, maximum )( rng );
}

void merge(int *c1, int *c2, int *p, int cl1, int cl2) {
	int i = 0, j = 0, k = 0;
	while(i < cl1 && j < cl2) {
		if(c1[i] < c2[j]) {
			p[k] = c1[i];
			i++;
			k++;
		}
		else {
			p[k] = c2[j];
			j++;
			k++;
		}
	}
	while(i < cl1) {
		p[k] = c1[i];
		i++;
		k++;
	}
	while(j < cl2) {
		p[k] = c2[j];
		j++;
		k++;
	}
}

void mergeSortSerial(int *arr, int length) {
	if(length <= 1)
		return;

	int mid = length / 2;
	
	int *left = new int[mid];
	int *right = new int[length-mid];

	for(int i = 0 ; i < mid ; i++) {
		left[i] = arr[i];
	}
	for(int i = mid ; i < length ; i++) {
		right[i-mid] = arr[i];
	}
	mergeSortSerial(left, mid);
	mergeSortSerial(right, length-mid);
	merge(left, right, arr, mid, length-mid);
}

void mergeSortParallel(int *arr, int length) {
	if(length <= 1)
		return;
	int mid = length / 2;
	int *left, *right;
	left = new int[mid];
	right = new int[length-mid];

	#pragma omp task firstprivate(left)
	{
		
		for(int i = 0 ; i < mid ; i++) {
			left[i] = arr[i];
		}
		mergeSortParallel(left, mid);
	}
	
	
	#pragma omp task firstprivate(right)
	{
		
		for(int i = mid ; i < length ; i++) {
			right[i-mid] = arr[i];
		}
	
		mergeSortParallel(right, length-mid);		
	}


	// #pragma omp parallel sections
	// {
	// 	#pragma omp section
	// 	{
	// 		left = new int[mid];
	// 		for(int i = 0 ; i < mid ; i++) {
	// 			left[i] = arr[i];
	// 		}
	// 		mergeSortParallel(left, mid);
	// 	}
	// 	#pragma omp section
	// 	{
	// 		right = new int[length-mid];
	// 		for(int i = mid ; i < length ; i++) {
	// 			right[i-mid] = arr[i];
	// 		}
	// 		mergeSortParallel(right, length-mid);
	// 	}
	// }
	#pragma omp taskwait
	merge(left, right, arr, mid, length-mid);
}

void init_array(int *arr1, int *arr2, int n) {
	for(int i = 0 ; i < n ; i++) {
		arr1[i] = arr2[i] = random_in_range(10,99);
	}
}

void print_array(int *arr, int n) {
	for(int i = 0 ; i < n ; i++) {
		cout<<arr[i]<<" ";
	}
}

int main() {
	int n = 2500;
	int *a, *b;
	a = new int[n];
	b = new int[n];

	init_array(a, b, n);
	cout<<"Initial vector: "<<endl;
	print_array(a, n);
	cout<<endl;
	cout<<endl;

	time_point<system_clock> starttime, endtime;
	starttime = system_clock::now();
	mergeSortSerial(a, n);
	endtime = system_clock::now();
	duration<double> time = endtime - starttime;
	cout<<"Time for serial: "<<1000*time.count()<<" milliseconds"<<endl;
	cout<<"Result after serial merge sort: "<<endl;
	print_array(a, n);
	cout<<endl;
	cout<<endl;


	starttime = system_clock::now();
	mergeSortParallel(b, n);
	endtime = system_clock::now();
	time = endtime - starttime;
	cout<<"Time for parallel: "<<1000*time.count()<<" milliseconds"<<endl;
	cout<<"Result after parallel merge sort: "<<endl;
	print_array(b, n);

	return 0;	
}