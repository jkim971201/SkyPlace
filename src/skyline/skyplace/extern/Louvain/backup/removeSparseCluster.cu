#include "modularity_optimisation.cuh"
#include "removeSparseCluster.cuh"
#include <thrust/scan.h>
#include <thrust/partition.h>
#include <vector>

namespace Louvain
{

struct checkSparseCluster
{
  checkSparseCluster(int minSize, int* communitySize)
  {
    minSize_       = minSize;
    communitySize_ = communitySize;
  }

  int  minSize_;
  int* communitySize_;

  __host__ __device__
  bool operator()(const int& communityIdx) const
  {
    return communitySize_[communityIdx] < minSize_;
  }
};

__global__ void computeEdgesSumForRemoveSparsity(device_structures deviceStructures) 
{
  int verticesPerBlock = blockDim.y;
  int concurrentNeighbours = blockDim.x;
  float edgesSum = 0;
  int vertex = blockIdx.x * verticesPerBlock + threadIdx.y;

  if(vertex < *deviceStructures.V) 
  {
    int startOffset = deviceStructures.edgesIndex[vertex], endOffset = deviceStructures.edgesIndex[vertex + 1];

    for(int index = startOffset + threadIdx.x; index < endOffset; index += concurrentNeighbours)
      edgesSum += deviceStructures.weights[index];

    for(int offset = concurrentNeighbours / 2; offset > 0; offset /= 2)
      edgesSum += __shfl_down_sync(FULL_MASK, edgesSum, offset);

    if(threadIdx.x == 0)
      deviceStructures.vertexEdgesSum[vertex] = edgesSum;
  }
}

__global__ void countCommunitySize(const int  originalV, 
                                         int* communitySize, 
                                   const int* originalToCommunity)
{
	// ID of original vertex
  int vID = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

  if(vID < originalV) 
  {
    int cID = originalToCommunity[vID];
    atomicAdd(&communitySize[cID], 1);
  }
}

__device__ float computeGainSparse(const int    vertex, 
                                   const int    community, 
                                   const int    currentCommunity, 
                                   const float* communityWeight,
                                   const float* vertexEdgesSum, 
                                   const float  vertexToCommunity) 
{
  float communitySum        = communityWeight[community];
  float currentCommunitySum = communityWeight[currentCommunity] - vertexEdgesSum[vertex];

  float gain = vertexToCommunity / M 
             + vertexEdgesSum[vertex] * (currentCommunitySum - communitySum) / (2 * M * M);
  return gain;
}


__device__ void computeMoveSparse(const int    numSparseCluster, 
                                  const int*   sparseClusterList, 
                                        device_structures deviceStructures)
{
  int clusterID = blockIdx.x * blockDim.x + threadIdx.y;

  if(clusterID < numSparseCluster) 
  {
		int  sparseClusterID    = sparseClusterList[clusterID];
    int* cluster2Community  = deviceStructures.vertexCommunity;
    int* edgesIndex         = deviceStructures.edgesIndex;
    int* edges              = deviceStructures.edges;
    int* clusterSize        = deviceStructures.communitySize;
    int* newClusterID       = deviceStructures.newVertexCommunity;
    float* weights          = deviceStructures.weights;
    float* communityWeight  = deviceStructures.communityWeight;
    float* clusterEdgesSum  = deviceStructures.vertexEdgesSum;

    int currentCommunityID  = cluster2Community[sparseClusterID];
    int bestCommunityID     = currentCommunityID;

    float bestGain          = -10.0;
    int incidentEdgeID      = edgesIndex[sparseClusterID]; // Start ID
    int upperBound          = edgesIndex[sparseClusterID + 1]; 

    // Visit all incident edges in the parallel manner
    // concurrent thread = blockdim.x
    while(incidentEdgeID < upperBound) // This is ok because the edge IDs are in the sorted order
    {
			int incidentClusterID   = edges[incidentEdgeID];
			int incidentCommunityID = cluster2Community[incidentClusterID];
			float edgeWeight        = weights[incidentEdgeID];

			float deltaWeight = 0.0;

			float gain = computeGainSparse(sparseClusterID, 
                                     incidentCommunityID, // community j
                                     currentCommunityID,  // community i
                                     communityWeight,     // float array : sum of internal edge weight of community c (a_c)
                                     clusterEdgesSum,     // float : sum of edge weight connected to cluster (vertex) i (k_i)
                                     deltaWeight);        // delta weight of moving to otherCommunity

			if(bestGain < gain)
				bestCommunityID = incidentCommunityID;

			incidentEdgeID += 1; 
			// Since we are visting incident clusters sequentially, not parallel, just add 1.
    }

    if(threadIdx.x == 0)
      newCommunityID[sparseClusterID] = bestCommunityID;
    else 
      newCommunityID[sparseClusterID] = currentCommunityID;
  }
}

void removeSparseCluster(int V,
                         int minNumVertex, 
                         host_structures& hostStructures, 
                       device_structures& deviceStructures)
{
	int originalV = hostStructures.originalV;
  int blocks    = (originalV + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  printf("Start removeSparseCluster...\n");

  computeEdgesSumForRemoveSparsity<<<blocksNumber(V, WARP_SIZE), 
                                     dim3{WARP_SIZE, THREADS_PER_BLOCK / WARP_SIZE}>>>(deviceStructures);

  HANDLE_ERROR( cudaMemcpy(hostStructures.edgesIndex, 
                         deviceStructures.edgesIndex,
                         (V + 1) * sizeof(int), 
                         cudaMemcpyDeviceToHost) );

	// Initialize
  thrust::fill(thrust::device, 
			         deviceStructures.communitySize, 
			         deviceStructures.communitySize + V, 0);

  countCommunitySize<<<blocks, THREADS_PER_BLOCK>>>(originalV, 
                                                    deviceStructures.communitySize, 
                                                    deviceStructures.originalToCommunity);

  int* partition = deviceStructures.partition;
  thrust::sequence(thrust::device, partition, partition + V, 0);
 
	                                                           // it has to be a device array???
  auto predicate          = checkSparseCluster(minNumVertex, deviceStructures.communitySize);
  int* deviceVerticesEnd  = thrust::partition(thrust::device, partition, partition + V, predicate);
  int  numSparseCluster   = thrust::distance(partition, deviceVerticesEnd);

  printf("Number of Sparse Cluster %d / %d \n", numSparseCluster, V);

  thrust::fill(thrust::device, 
			         deviceStructures.communitySize, 
			         deviceStructures.communitySize + V, 1.0);

	if(numSparseCluster > 0)
	{
		int numThreadCluster = 32;
		int numBlockCluster  = (numSparseCluster + numThreadCluster - 1)
			                   / numThreadCluster;

    computeMoveSparse<<<numBlockCluster, numThreadCluster>>>(numSparseCluster, 
                                                             partition, 
                                                             deviceStructures);
	}
}

};
