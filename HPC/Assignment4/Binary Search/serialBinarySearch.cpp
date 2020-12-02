#include<iostream>
#include<cstdlib>
#include<math.h>
#include<chrono>
#include<random>
#include <bits/stdc++.h>

using namespace std;
using namespace std::chrono;

void binarySearch(int *arr, int start, int end, int key)
{
    while(start<=end)
    {
        int mid=(start+end)/2;
        if(arr[mid]==key)
        {
            cout<<"Element is Found"<<endl;
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
    int n = 1000;
    int *arr = new int[n];

    initializeArray(arr,n);
    sort(arr, arr + n);

    cout<<"Random Initialized Array:"<<endl;
    for(int i = 0 ; i < n ; i++) {
        cout<<arr[i]<<" ";
    }
    cout<<endl;
    
    int keyIndex = getRandomKey();
    int key = arr[keyIndex];
    cout<<"Element to search: "<<key<<endl;

    time_point<system_clock> starttime, endtime;
    starttime = system_clock::now();
    binarySearch(arr,0,999,key);
    endtime = system_clock::now();
    duration <double> time= endtime - starttime;

    cout<<"Time for serial: "<<1000*time.count()<<" milliseconds"<<endl;
    return 0;
}