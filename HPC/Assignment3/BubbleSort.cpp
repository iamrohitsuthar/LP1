#include <iostream>
#include <chrono>
#include <omp.h>
#include <stdio.h>
#include <random>

using namespace std;
using namespace std::chrono;

int random_in_range( int minimum, int maximum )
{
  thread_local std::ranlux48 rng( 
    std::chrono::system_clock::now().time_since_epoch().count() );
  return std::uniform_int_distribution <int> ( minimum, maximum )( rng );
}


void bubbleSortSerial(int a[], int n)
{
	time_point<system_clock> starttime, endtime;
	starttime = system_clock::now();

	for (int i = 0; i < n-1; i++)
	{
		for (int j = 0; j < n-1; j++)
		{
			if(a[j] > a[j+1])
			{
				int temp = a[j];
				a[j] = a[j+1];
				a[j+1] = temp;
			}
		}
	}
	endtime = system_clock::now();
	duration <double> time= endtime - starttime;
	cout<<"Time for serial: "<<1000*time.count()<<" milliseconds"<<endl;
}

void bubbleSortOddEven(int b[], int n)
{
	time_point<system_clock> starttime, endtime;
	starttime = system_clock::now();
	int pass;

	for(int i = 0 ; i < n-1 ; i++)
	{
		pass = i % 2;
		for (int j = pass ; j < n-1 ; j+=2)
		{
			if(b[j]>b[j+1])
			{
				int temp = b[j];
				b[j] = b[j+1];
				b[j+1]=temp;	
			}
		}
	}
	endtime = system_clock::now();
	duration<double> time = endtime - starttime;
	cout<<"Time for Bubble sort (Odd Even Transposition): "<<1000*time.count()<<" milliseconds"<<endl;
}

void bubbleSortParallel(int b[], int n)
{
	time_point<system_clock> starttime, endtime;
	starttime = system_clock::now();
	int pass;

	omp_set_num_threads(2);

	for(int i = 0 ; i < n-1 ; i++)
	{
		pass = i % 2;
		#pragma omp parallel for default(none), shared(b,first,n)
		for (int j = pass ; j < n-1 ; j+=2)
		{
			if(b[j]>b[j+1])
			{
				int temp = b[j];
				b[j] = b[j+1];
				b[j+1]=temp;	
			}
		}
	}
	endtime = system_clock::now();
	duration<double> time = endtime - starttime;
	cout<<"Time for Parallel: "<<1000*time.count()<<" milliseconds"<<endl;
}

void init_array(int *arr1, int *arr2, int *arr3, int n) {
	for(int i = 0 ; i < n ; i++) {
		arr1[i] = arr2[i] = arr3[i] = random_in_range(10,9999);
	}
}

void print_array(int *arr, int n) {
	for(int i = 0 ; i < n ; i++) {
		cout<<arr[i]<<" ";
	}
}

int main()
{
	int n = 5000;
	int *a, *b, *c;
	a = new int[n];
	b = new int[n];
	c = new int[n];
	init_array(a, b, c, n);
	cout<<"Initial vector: "<<endl;
	print_array(a, n);
	cout<<endl;
	cout<<endl;

	bubbleSortSerial(a,n);
	cout<<"Result after serial bubble sort: "<<endl;
	print_array(a, n);

	cout<<endl;
	cout<<endl;

	bubbleSortOddEven(c,n);
	cout<<"Result after odd-even sort: "<<endl;
	print_array(c, n);

	cout<<endl;
	cout<<endl;
	bubbleSortParallel(b,n);
	cout<<"Result after parallel bubble sort: "<<endl;
	print_array(b, n);
	cout<<endl;

	return 0;
}