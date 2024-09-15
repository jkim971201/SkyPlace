#include "utils.cuh"
#include <vector>
#include <iostream>
#include <thrust/partition.h>
#include <fstream>
#include <sstream>

namespace Louvain
{

void copyStructures(host_structures& hostStructures, device_structures& deviceStructures,
          aggregation_phase_structures& aggregationPhaseStructures) {
  // copying from deviceStructures to hostStructures
  int V = hostStructures.V, E = hostStructures.E;
  HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.vertexCommunity, V * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.communityWeight, V * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.edges, E * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.weights, E * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.edgesIndex, (V + 1) * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.originalToCommunity, V * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.vertexEdgesSum, V * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.newVertexCommunity, V * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.V, sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.E, sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.originalV, sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.communitySize, V * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.partition, V * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&deviceStructures.toOwnCommunity, V * sizeof(int)));

  thrust::fill(thrust::device, deviceStructures.communitySize, deviceStructures.communitySize + V, 1);
  thrust::sequence(thrust::device, deviceStructures.vertexCommunity, deviceStructures.vertexCommunity + V, 0);
  thrust::sequence(thrust::device, deviceStructures.newVertexCommunity, deviceStructures.newVertexCommunity + V, 0);
  thrust::sequence(thrust::device, deviceStructures.originalToCommunity, deviceStructures.originalToCommunity + V, 0);

  HANDLE_ERROR(cudaMemcpy(deviceStructures.communityWeight, hostStructures.communityWeight, V * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(deviceStructures.edges, hostStructures.edges, E * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(deviceStructures.weights, hostStructures.weights, E * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(deviceStructures.edgesIndex, hostStructures.edgesIndex, (V + 1) * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(deviceStructures.V, &hostStructures.V, sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(deviceStructures.E, &hostStructures.E, sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(deviceStructures.originalV, &hostStructures.originalV, sizeof(int), cudaMemcpyHostToDevice));

  // preparing aggregationPhaseStructures
  HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.communityDegree, V * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.newID, V * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.edgePos, V * sizeof(int)));;
  HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.vertexStart, V * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.orderedVertices, V * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.edgeIndexToCurPos, E * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.newEdges, E * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&aggregationPhaseStructures.newWeights, E * sizeof(float)));
}

void deleteStructures(device_structures& deviceStructures,
           aggregation_phase_structures& aggregationPhaseStructures) 
{
  HANDLE_ERROR(cudaFree(deviceStructures.originalV));
  HANDLE_ERROR(cudaFree(deviceStructures.vertexCommunity));
  HANDLE_ERROR(cudaFree(deviceStructures.communityWeight));
  HANDLE_ERROR(cudaFree(deviceStructures.edges));
  HANDLE_ERROR(cudaFree(deviceStructures.weights));
  HANDLE_ERROR(cudaFree(deviceStructures.edgesIndex));
  HANDLE_ERROR(cudaFree(deviceStructures.originalToCommunity));
  HANDLE_ERROR(cudaFree(deviceStructures.vertexEdgesSum));
  HANDLE_ERROR(cudaFree(deviceStructures.newVertexCommunity));
  HANDLE_ERROR(cudaFree(deviceStructures.E));
  HANDLE_ERROR(cudaFree(deviceStructures.V));
  HANDLE_ERROR(cudaFree(deviceStructures.communitySize));
  HANDLE_ERROR(cudaFree(deviceStructures.partition));
  HANDLE_ERROR(cudaFree(deviceStructures.toOwnCommunity));

  HANDLE_ERROR(cudaFree(aggregationPhaseStructures.communityDegree));
  HANDLE_ERROR(cudaFree(aggregationPhaseStructures.newID));
  HANDLE_ERROR(cudaFree(aggregationPhaseStructures.edgePos));
  HANDLE_ERROR(cudaFree(aggregationPhaseStructures.vertexStart));
  HANDLE_ERROR(cudaFree(aggregationPhaseStructures.orderedVertices));
  HANDLE_ERROR(cudaFree(aggregationPhaseStructures.edgeIndexToCurPos));
  HANDLE_ERROR(cudaFree(aggregationPhaseStructures.newEdges));
  HANDLE_ERROR(cudaFree(aggregationPhaseStructures.newWeights));
}

int blocksNumber(int V, int threadsPerVertex) 
{
  return (V * threadsPerVertex + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

bool isPrime(int n) 
{
  for (int i = 2; i < sqrt(n) + 1; i++)
    if (n % i == 0)
      return false;
  return true;
}

int getPrime(int n) 
{
  do {
    n++;
  } while(!isPrime(n));
  return n;
}

};
