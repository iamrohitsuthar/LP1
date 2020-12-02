#include<iostream>
#include<cstdlib>
#include<math.h>
#include<chrono>
#include<random>
#include <bits/stdc++.h>

using namespace std;
using namespace std::chrono;

void BFS(int start, int no_of_vertices, int adj[100][100]) 
{ 
    vector<bool> visited(no_of_vertices, false); 
    vector<int> q; 
    q.push_back(start); 

    visited[start] = true; 

    int vis; 
    while (!q.empty()) { 
        vis = q[0]; 

        cout << vis << " "; 
        q.erase(q.begin()); 

        for (int i = 0; i < no_of_vertices; i++) { 
            if (adj[vis][i] == 1 && (!visited[i])) { 
                q.push_back(i);
                visited[i] = true; 
            } 
        } 
    } 
} 

int main() 
{ 
    int no_of_vertices, source_vertex;
    int adjacency_matrix[100][100];

    cout<<"Enter the number of vertices\n";
    cin>>no_of_vertices;

    cout<<"Enter the Adjacency Matrix\n";
    for(int i = 0; i < no_of_vertices ; i++)
    {
        for(int j = 0 ; j < no_of_vertices ; j++) {
            cin>>adjacency_matrix[i][j];
        }
    }

    cout<<"Enter the Source Vertex\n";
    cin>>source_vertex;

    cout<<"BFS Traversal: ";

    time_point<system_clock> starttime, endtime;
    starttime = system_clock::now();
    BFS(source_vertex, no_of_vertices, adjacency_matrix); 
    endtime = system_clock::now();
    duration <double> time= endtime - starttime;

    cout<<endl;
    cout<<"Time for serial: "<<1000*time.count()<<" milliseconds"<<endl;
}