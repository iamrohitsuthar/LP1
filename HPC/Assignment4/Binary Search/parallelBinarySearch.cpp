#include<iostream>
#include<mpi.h>
#include<cstdlib>
#include<math.h>
#include<chrono>
#include<random>
#include <bits/stdc++.h>
using namespace std;

void binarySearch(int *arr, int start, int end, int key, int rank)
{
    while(start<=end)
    {
        int mid=(start+end)/2;
        if(arr[mid]==key)
        {
            cout<<"Element found by processor rank "<<rank<< " at index " <<mid<<endl;
            return;
        }
        else if(arr[mid]<key)
        {
            start=mid+1;
        }
        else
        {
            end=mid-1;
        }
    }
}

int random_in_range( int minimum, int maximum )
{
  thread_local std::ranlux48 rng( 
    std::chrono::system_clock::now().time_since_epoch().count() );
  return std::uniform_int_distribution <int> ( minimum, maximum )( rng );
}

void initializeArray(int *arr, int n) {
    for(int i = 0 ; i < n ; i++) {
        arr[i] = random_in_range(1000,9999);
    }
}

int getRandomKey() {
    return random_in_range(0,1000);   
}

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 1000;
    int *arr = new int[n];
    int key;

    int blocks = 2;
	int blockSize = n / blocks;

    if(rank == 0) {
    	arr = new int[n];

    	initializeArray(arr,n);
    	sort(arr, arr + n);
    	MPI_Send(arr, n, MPI_INT, 1, 0, MPI_COMM_WORLD);
    	int keyIndex = getRandomKey();
	    key = arr[keyIndex];
	    MPI_Send(&key, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
 
	    cout<<"Random Initialized Array:"<<endl;

	    for(int i = 0 ; i < n ; i++) {
	        cout<<arr[i]<<" ";
	    }
	    cout<<endl;
	  
	    cout<<"Element to search: "<<key<<endl;

	    cout<<"Processor rank: "<<rank<<"\nSize : "<<size<<endl;

	    double start = MPI_Wtime();
	    binarySearch(arr, rank * blockSize, (rank+1) * blockSize - 1, key, rank);
	    double end = MPI_Wtime();

	    cout<<"Execution time of Processor "<<rank<<" is "<<(end-start)*1000<<endl;
	}
	else if(rank == 1) {
		MPI_Recv(arr, n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&key, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		cout<<"Processor rank: "<<rank<<"\nSize : "<<size<<endl;
	    
	    double start = MPI_Wtime();
	    binarySearch(arr, rank * blockSize, (rank+1) * blockSize - 1, key, rank);
	    double end = MPI_Wtime();

	    cout<<"Execution time of Processor "<<rank<<" is "<<(end-start)*1000<<endl;
	}

    MPI_Finalize();
    return 0;
}