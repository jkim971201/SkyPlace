#include "community_aggregation.cuh"
#include <thrust/scan.h>
#include <thrust/partition.h>

namespace Louvain
{

/**
 * Computes hash (using double hashing) for open-addressing purposes of arrays in prepareHashArrays function.
 * @param val   value we want to insert
 * @param index current position
 * @param prime size of hash array
 * @return hash
 */
__device__ int getHashAggregation(int val, int index, int prime) {
  int h1 = val % prime;
  int h2 = 1 + (val % (prime - 1));
  return (h1 + index * h2) % prime;
}

/**
 * Fills content of hashCommunity and hashWeights arrays that are later used in mergeCommunity function.
 * @param community        neighbour's community
 * @param prime            prime number used for hashing
 * @param weight           neighbour's weight
 * @param hashWeight       table of sum of weights between vertices and communities
 * @param hashCommunity    table informing which community's info is stored in given index
 * @param hashTablesOffset offset of the vertex in hash arrays (single hash array may contain multiple vertices)
 * @return curPos, if this was first addition, -1 otherwise
 */
__device__ int prepareHashArraysAggregation(const int    community, 
                                            const int    prime, 
                                            const float  weight,
                                                  float* hashWeight, 
                                                  int*   hashCommunity,    
                                            const int    hashTablesOffset) 
{
  int it = 0;
  while(true) 
  {
    int curPos = hashTablesOffset + getHashAggregation(community, it++, prime);
    if(hashCommunity[curPos] == community) 
    {
      atomicAdd(&hashWeight[curPos], weight);
      return -1;
    } 
    else if(hashCommunity[curPos] == -1) 
    {
      if(atomicCAS(&hashCommunity[curPos], -1, community) == -1) 
      {
        atomicAdd(&hashWeight[curPos], weight);
        return curPos;
      } 
      else if(hashCommunity[curPos] == community) 
      {
        atomicAdd(&hashWeight[curPos], weight);
        return -1;
      }
    }
  }
}

__global__ void fillArrays(const int  V, 
                                 int* communitySize, 
                                 int* communityDegree, 
                                 int* newID, 
                           const int* vertexCommunity, 
                           const int* edgesIndex) 
{
  int vertex = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

  // i
  if(vertex < V) 
  {
    int community = vertexCommunity[vertex]; // c[i]
    atomicAdd(&communitySize[community], 1);
    int vertexDegree = edgesIndex[vertex + 1] - edgesIndex[vertex];
    atomicAdd(&communityDegree[community], vertexDegree);
    newID[community] = 1;
  }
}

/**
 * orderVertices is responsible for generating ordered (meaning vertices in the same community are placed
 * next to each other) vertices.
 * @param V               - number of vertices
 * @param orderedVertices - ordered vertices
 * @param vertexStart     - community -> begin index in orderedVertices array
 *                          NOTE: atomicAdd changes values in this array, that's why it has to be reset afterwards
 * @param vertexCommunity - vertex -> community
 */
__global__ void orderVertices(const int  V, 
                                    int* orderedVertices, 
                                    int* vertexStart, 
                              const int* vertexCommunity) 
{
  int vertex = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

  if(vertex < V) 
  {
    int community = vertexCommunity[vertex];
    int index = atomicAdd(&vertexStart[community], 1);
    orderedVertices[index] = vertex;
  }
}

__device__ void mergeCommunity(const int    V, 
                               const int*   communities, // Communities that are in the degree range
                                     device_structures deviceStructures, 
                               const int    prime, 
                               const int*   edgePos,
                                     int*   communityDegree, 
                               const int*   orderedVertices, 
                               const int*   vertexStart, 
                                     int*   edgeIndexToCurPos, 
                                     int*   newEdges,
                                     float* newWeights, 
                                     int*   hashCommunity, 
                                     float* hashWeight, 
                                     int*   prefixSum) 
{
  int communitiesOwned    = 0;
  int communitiesPerBlock = blockDim.y;
  int concurrentThreads   = blockDim.x;
  int hashTablesOffset    = threadIdx.y * prime;
  int communityIndex      = blockIdx.x * communitiesPerBlock + threadIdx.y;

  if(communityIndex < V) 
  {
    int community = communities[communityIndex];

    if(deviceStructures.communitySize[community] > 0) 
    {
      for(unsigned int i = threadIdx.x; i < prime; i += concurrentThreads) 
      {
        hashWeight[hashTablesOffset + i] = 0;
        hashCommunity[hashTablesOffset + i] = -1;
      }

      if(concurrentThreads > WARP_SIZE)
        prefixSum[threadIdx.x] = 0;

      if(concurrentThreads > WARP_SIZE)
        __syncthreads();

      // Filling hash tables content for every vertex in community
      for(int vertexIndex = 0; vertexIndex < deviceStructures.communitySize[community]; vertexIndex++) 
      {
        int vertex          = orderedVertices[vertexStart[community] + vertexIndex];
        int vertexBaseIndex = deviceStructures.edgesIndex[vertex];
        int vertexDegree    = deviceStructures.edgesIndex[vertex + 1] - vertexBaseIndex;

        for(int neighbourIndex = threadIdx.x; neighbourIndex < vertexDegree; neighbourIndex += concurrentThreads) 
        {
          int index     = vertexBaseIndex + neighbourIndex;
          int neighbour = deviceStructures.edges[index];
          float weight  = deviceStructures.weights[index];
          int neighbourCommunity = deviceStructures.vertexCommunity[neighbour];
          int curPos = prepareHashArraysAggregation(neighbourCommunity, 
                                                    prime, 
                                                    weight, 
                                                    hashWeight,    // changed
                                                    hashCommunity, // changed
                                                    hashTablesOffset);

          if(curPos > -1) 
          {
            edgeIndexToCurPos[index] = curPos;
            communitiesOwned++;
          }
        }
      }

      int communitiesOwnedPrefixSum = communitiesOwned;
      if(concurrentThreads <= WARP_SIZE) 
      {
        for(unsigned int offset = 1; offset <= concurrentThreads / 2; offset *= 2) 
        {
          int otherSum = __shfl_up_sync(FULL_MASK, communitiesOwnedPrefixSum, offset);
          if(threadIdx.x >= offset) 
            communitiesOwnedPrefixSum += otherSum;
        }
        // subtraction to have exclusive sum
        communitiesOwnedPrefixSum -= communitiesOwned;
      } 
      else 
      {
        for(unsigned int offset = 1; offset <= concurrentThreads / 2; offset *= 2) 
        {
          __syncthreads();
          prefixSum[threadIdx.x] = communitiesOwnedPrefixSum;
          __syncthreads();
          if(threadIdx.x >= offset)
            communitiesOwnedPrefixSum += prefixSum[threadIdx.x - offset];
        }
        // subtraction to have exclusive sum
        communitiesOwnedPrefixSum -= communitiesOwned;
      }

      int newEdgesIndex = edgePos[community] + communitiesOwnedPrefixSum;
      if(threadIdx.x == concurrentThreads - 1) 
      {
        communityDegree[community] = communitiesOwnedPrefixSum + communitiesOwned;
        atomicAdd(deviceStructures.E, communityDegree[community]);
      }
       
      for(int vertexIndex = 0; vertexIndex < deviceStructures.communitySize[community]; vertexIndex++) 
      {
        int vertex = orderedVertices[vertexStart[community] + vertexIndex];
        int vertexBaseIndex = deviceStructures.edgesIndex[vertex];
        int vertexDegree = deviceStructures.edgesIndex[vertex + 1] - vertexBaseIndex;

        for(int neighbourIndex = threadIdx.x; neighbourIndex < vertexDegree; neighbourIndex += concurrentThreads) 
        {
          int index  = vertexBaseIndex + neighbourIndex;
          int curPos = edgeIndexToCurPos[index];

          if(curPos > -1) 
          {
            newEdges[newEdgesIndex]   = hashCommunity[curPos];
            newWeights[newEdgesIndex] = hashWeight[curPos];
            newEdgesIndex++;
          }
        }
      }
    }
  }
}

__global__ void mergeCommunityShared(const int    V, 
                                     const int*   communities, 
                                           device_structures deviceStructures, 
                                     const int    prime, 
                                     const int*   edgePos,
                                           int*   communityDegree, 
                                     const int*   orderedVertices, 
                                     const int*   vertexStart,
                                           int*   edgeIndexToCurPos, 
                                           int*   newEdges,
                                           float* newWeights) 
{
  int communitiesPerBlock = blockDim.y;
  int communityIndex      = blockIdx.x * communitiesPerBlock + threadIdx.y;

  if(communityIndex < V) 
  {
    extern __shared__ int s[];
    int* hashCommunity = s;
    auto *hashWeight = (float*) &hashCommunity[communitiesPerBlock * prime];
    auto *prefixSum  = (int*)   &hashWeight[communitiesPerBlock * prime];

    mergeCommunity(V, 
                   communities, 
                   deviceStructures, 
                   prime, 
                   edgePos, 
                   communityDegree, 
                   orderedVertices, 
                   vertexStart,
                   edgeIndexToCurPos, 
                   newEdges, 
                   newWeights, 
                   hashCommunity, 
                   hashWeight, 
                   prefixSum);
  }
}

__global__ void mergeCommunityGlobal(const int    V, 
                                     const int*   communities, 
                                           device_structures deviceStructures, 
                                     const int    prime, 
                                     const int*   edgePos,
                                           int*   communityDegree, 
                                     const int*   orderedVertices, 
                                     const int*   vertexStart,
                                           int*   edgeIndexToCurPos, 
                                           int*   newEdges,
                                           float* newWeights, 
                                           int*   hashCommunity, 
                                           float* hashWeight) 
{
  int communitiesPerBlock = blockDim.y;
  int communityIndex = blockIdx.x * communitiesPerBlock + threadIdx.y;

  if(communityIndex < V) 
  {
    extern __shared__ int s[];
    auto *prefixSum = s;

    hashCommunity = &hashCommunity[blockIdx.x * prime];
    hashWeight    = &hashWeight[blockIdx.x * prime];

    mergeCommunity(V, 
                   communities, 
                   deviceStructures, 
                   prime, 
                   edgePos, 
                   communityDegree, 
                   orderedVertices, 
                   vertexStart,
                   edgeIndexToCurPos, 
                   newEdges,
                   newWeights, 
                   hashCommunity, 
                   hashWeight, 
                   prefixSum);
  }
}

__global__ void compressEdges(const int    V, 
                                    device_structures deviceStructures, 
                              const int*   communityDegree, 
                              const int*   newEdges,
                              const float* newWeights, 
                              const int*   newID, 
                              const int*   edgePos, 
                              const int*   vertexStart) 
{
  int communitiesPerBlock = blockDim.y;
  int concurrentThreads = blockDim.x;
  int community = blockIdx.x * communitiesPerBlock + threadIdx.y;

  if(blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) 
    deviceStructures.edgesIndex[*deviceStructures.V] = *deviceStructures.E;

  if(community < V && deviceStructures.communitySize[community] > 0) 
  {
    int neighboursBaseIndex = edgePos[community];
    int communityNewID      = newID[community];

    if(threadIdx.x == 0) 
    {
      deviceStructures.vertexCommunity[communityNewID]    = communityNewID;
      deviceStructures.newVertexCommunity[communityNewID] = communityNewID;
      deviceStructures.edgesIndex[communityNewID]         = vertexStart[community];
    }

    for(int neighbourIndex = threadIdx.x; neighbourIndex < communityDegree[community]; neighbourIndex += concurrentThreads) 
    {
      int newIndex = neighbourIndex + neighboursBaseIndex;
      int oldIndex = vertexStart[community] + neighbourIndex;
      deviceStructures.edges[oldIndex] = newID[newEdges[newIndex]];
      deviceStructures.weights[oldIndex] = newWeights[newIndex];
      atomicAdd(&deviceStructures.communityWeight[communityNewID], newWeights[newIndex]);
    }
  }
}

__global__ void updateOriginalToCommunity(device_structures deviceStructures, 
                                          const int *newID) 
{
  int vertex = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  if (vertex < *deviceStructures.originalV) 
  {
    int community = deviceStructures.originalToCommunity[vertex];
    deviceStructures.originalToCommunity[vertex] = newID[community];
  }
}

struct IsInBucketAggregation
{
  IsInBucketAggregation(int llowerBound, int uupperBound, int *ccomunityDegree) 
  {
    lowerBound = llowerBound;
    upperBound = uupperBound;
    communityDegree = ccomunityDegree;
  }

  int lowerBound, upperBound;
  int *communityDegree;
  __host__ __device__
  bool operator()(const int &v) const
  {
    int edgesNumber = communityDegree[v];
    return edgesNumber > lowerBound && edgesNumber <= upperBound;
  }
};

void aggregateCommunities(device_structures& deviceStructures, 
                            host_structures& hostStructures,
               aggregation_phase_structures& aggregationPhaseStructures) 
{
  int V = hostStructures.V, E = hostStructures.E;
  int blocks = (V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  int*   communityDegree    = aggregationPhaseStructures.communityDegree;
  int*   newID              = aggregationPhaseStructures.newID;
  int*   edgePos            = aggregationPhaseStructures.edgePos;
  int*   vertexStart        = aggregationPhaseStructures.vertexStart;
  int*   orderedVertices    = aggregationPhaseStructures.orderedVertices;
  int*   edgeIndexToCurPos  = aggregationPhaseStructures.edgeIndexToCurPos;
  int*   newEdges           = aggregationPhaseStructures.newEdges;
  float* newWeights         = aggregationPhaseStructures.newWeights;

  //int vertices[V];

  int* vertices;
  HANDLE_ERROR(cudaHostAlloc((void**)&vertices, V * sizeof(int), cudaHostAllocDefault));

  for(int i = 0; i < V; i++)
    vertices[i] = i;

  int* deviceVertices;
  HANDLE_ERROR(cudaMalloc((void**)&deviceVertices, V * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(deviceVertices, vertices, V * sizeof(int), cudaMemcpyHostToDevice));

  thrust::fill(thrust::device, newID, newID + V, 0);
  thrust::fill(thrust::device, deviceStructures.communitySize, deviceStructures.communitySize + V, 0);
  thrust::fill(thrust::device, communityDegree, communityDegree + V, 0);

  // Algorithm 3 Line 4-6 (See Paper)
  fillArrays<<<blocks, THREADS_PER_BLOCK>>>(V, 
                                            deviceStructures.communitySize, 
                                            communityDegree, 
                                            newID,
                                            deviceStructures.vertexCommunity, 
                                            deviceStructures.edgesIndex);

  // Now we are ready to aggregate
  int newV = thrust::reduce(thrust::device, newID, newID + V);

  // TIP : 'Scan' of size or degree vector can make a vector of startID
  thrust::exclusive_scan(thrust::device, newID, newID + V , newID);
  thrust::exclusive_scan(thrust::device, communityDegree, communityDegree + V, edgePos);
  thrust::exclusive_scan(thrust::device, deviceStructures.communitySize, deviceStructures.communitySize + V, vertexStart);

  orderVertices<<<blocks, THREADS_PER_BLOCK>>>(V, orderedVertices, vertexStart, deviceStructures.vertexCommunity);
  // resetting vertexStart state to one before orderVertices call
  thrust::exclusive_scan(thrust::device, deviceStructures.communitySize, deviceStructures.communitySize + V, vertexStart);
  thrust::fill(thrust::device, edgeIndexToCurPos, edgeIndexToCurPos + E, -1);

  int bucketsSize = 4;
  int buckets[]   = {0, 127, 479, INT_MAX};
  int primes[]    = {191, 719};

  dim3 dims[] { {32, 4}, {128, 1}, {128, 1} };

  thrust::fill(thrust::device, deviceStructures.E, deviceStructures.E + 1, 0);

  // 1st Iter -> bucketNum =   0 -> only cares degree between (0,   127]
  // 2nd Iter -> bucketNum = 127 -> only cares degree between (127, 479]
  for(int bucketNum = 0; bucketNum < bucketsSize - 2; bucketNum++) 
  {
    dim3 blockDimension    = dims[bucketNum];
    int  prime             = primes[bucketNum];
    // IsInBucketAggregation returns true if communityDegree is between LB and UB
    auto predicate         = IsInBucketAggregation(buckets[bucketNum], buckets[bucketNum + 1], communityDegree);
    int* deviceVerticesEnd = thrust::partition(thrust::device, deviceVertices, deviceVertices + hostStructures.V, predicate);
    int  partitionSize     = thrust::distance(deviceVertices, deviceVerticesEnd);
    // thrust::distance returns the distance between two iterators

    // PartitionSize means the number of community members whose degree is in the current range
    if(partitionSize > 0) // If partition is not empty
    {
                                   // 1st Iter : 4    191
                                   // 2nd Iter : 1    719
      unsigned int sharedMemSize = blockDimension.y * prime * (sizeof(float) + sizeof(int));

      // 1st Iter :  32
      // 2nd Iter : 128
      if(blockDimension.x > WARP_SIZE) // WARP_SIZE := 32
        sharedMemSize += blockDimension.x * sizeof(int);
      
      unsigned int blocksDegrees = (partitionSize + blockDimension.y - 1) / blockDimension.y;

      // blockDegree    ~ partitionSize / numCommunityPerBlock
      // blockDimension ~ numCommunityPerBlock
      mergeCommunityShared<<<blocksDegrees, blockDimension, sharedMemSize>>>(partitionSize, 
                                                                             deviceVertices, 
                                                                             deviceStructures,  // E is changed
                                                                             prime, 
                                                                             edgePos,
                                                                             communityDegree,   // changed
                                                                             orderedVertices,  
                                                                             vertexStart, 
                                                                             edgeIndexToCurPos, // changed 
                                                                             newEdges,          // changed
                                                                             newWeights);       // changed
    }
  }

  dim3 blockDimension;
  // last bucket case
  int bucketNum = bucketsSize - 2; // bucketsSize := 4
  blockDimension = dims[bucketNum];
  int commDegree = newV;
  int prime = getPrime(commDegree * 1.5);
  auto predicate         = IsInBucketAggregation(buckets[bucketNum], buckets[bucketNum + 1], communityDegree);
  int* deviceVerticesEnd = thrust::partition(thrust::device, deviceVertices, deviceVertices + hostStructures.V, predicate);
  int  partitionSize     = thrust::distance(deviceVertices, deviceVerticesEnd);

  // For community whose degree is too big ( > 479 )
  if(partitionSize > 0) 
  {
    int*   hashCommunity;
    float* hashWeight;
    HANDLE_ERROR(cudaMalloc((void**)&hashCommunity, prime * partitionSize * sizeof(int)   ));
    HANDLE_ERROR(cudaMalloc((void**)&hashWeight,    prime * partitionSize * sizeof(float) ));

    unsigned int sharedMemSize = THREADS_PER_BLOCK * sizeof(int);
    unsigned int blocksDegrees = (partitionSize + blockDimension.y - 1) / blockDimension.y;

    mergeCommunityGlobal<<<blocksDegrees, blockDimension, sharedMemSize>>>(partitionSize, 
                                                                           deviceVertices, 
                                                                           deviceStructures, 
                                                                           prime, 
                                                                           edgePos,
                                                                           communityDegree, 
                                                                           orderedVertices, 
                                                                           vertexStart, 
                                                                           edgeIndexToCurPos, 
                                                                           newEdges, 
                                                                           newWeights,
                                                                           hashCommunity, 
                                                                           hashWeight);

    HANDLE_ERROR(cudaFree(hashCommunity));
    HANDLE_ERROR(cudaFree(hashWeight));
  }

  HANDLE_ERROR(cudaMemcpy(&hostStructures.E, deviceStructures.E, sizeof(int), cudaMemcpyDeviceToHost));
  hostStructures.V = newV;
  HANDLE_ERROR(cudaMemcpy(deviceStructures.V, &newV, sizeof(int), cudaMemcpyHostToDevice));
  thrust::fill(thrust::device, deviceStructures.communitySize, deviceStructures.communitySize + hostStructures.V, 1);
  int blocksNum = (V * WARP_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  blockDimension = {WARP_SIZE, THREADS_PER_BLOCK / WARP_SIZE};

  thrust::fill(thrust::device, deviceStructures.communityWeight, deviceStructures.communityWeight + hostStructures.V, 0.0);
  // vertexStart will contain starting indexes in compressed list
  thrust::exclusive_scan(thrust::device, communityDegree, communityDegree + V, vertexStart);
  compressEdges<<<blocksNum, blockDimension>>>(V, deviceStructures, communityDegree, newEdges, newWeights, newID, edgePos, vertexStart);
  HANDLE_ERROR(cudaFree(deviceVertices));
  updateOriginalToCommunity<<<(hostStructures.originalV + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(deviceStructures, newID);

  HANDLE_ERROR(cudaFreeHost(vertices));
}

};
