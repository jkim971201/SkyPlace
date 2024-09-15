#include "modularity_optimisation.cuh"
#include <thrust/partition.h>
#include <vector>

namespace Louvain
{

/**
 * Computes hashing (using double hashing) for open-addressing purposes of arrays in prepareHashArrays function.
 * @param val   value we want to insert
 * @param index current position
 * @param prime size of hash array
 * @return hash
 */
__device__ int getHash(int val, int index, int prime) 
{
  int h1 = val % prime;
  int h2 = 1 + (val % (prime - 1));
  return (h1 + index * h2) % prime;
}

/**
 * Computes sum of weights of edges adjacent to vertices (results are stored in vertexEdgesSum).
 * @param deviceStructures structures stored in device memory
 */
__global__ void computeEdgesSum(device_structures deviceStructures) 
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

/**
 * Computes sum of weights of edges adjacent to vertices (results are stored in vertexEdgesSum).
 * @param V               number of vertices
 * @param communityWeight community -> weight (sum of edges adjacent to vertices of community)
 * @param vertexCommunity vertex -> community assignment
 * @param vertexEdgesSum  vertex -> sum of edges adjacent to vertex
 */
__global__ void computeCommunityWeight(device_structures deviceStructures) 
{
  int vertex = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

  if(vertex < *deviceStructures.V) 
  {
    int community = deviceStructures.vertexCommunity[vertex];
    atomicAdd(&deviceStructures.communityWeight[community], deviceStructures.vertexEdgesSum[vertex]);
  }
}

/**
 * Fills content of hashCommunity and hash_weights arrays that are later used in computeGain function.
 * @param community        neighbour's community
 * @param prime            prime number used for hashing
 * @param weight           neighbour's weight
 * @param hashWeight       table of sum of weights between vertices and communities
 * @param hashCommunity    table informing which community's info is stored in given index
 * @param hashTablesOffset offset of the vertex in hash arrays (single hash array may contain multiple vertices)
 */
__device__ int prepareHashArrays(const int    community, 
                                 const int    prime, 
                                 const float  weight, 
                                       float* hashWeight, 
                                       int*   hashCommunity,
                                 const int    hashTablesOffset) 
{
  int it = 0;
  int curPos;

  do {
    curPos = hashTablesOffset + getHash(community, it++, prime);
    if(hashCommunity[curPos] == community)
      atomicAdd(&hashWeight[curPos], weight);
    else if(hashCommunity[curPos] == -1)
    {
      if(atomicCAS(&hashCommunity[curPos], -1, community) == -1)
        atomicAdd(&hashWeight[curPos], weight);
      else if(hashCommunity[curPos] == community)
        atomicAdd(&hashWeight[curPos], weight);
    }
  } while(hashCommunity[curPos] != community);

  return curPos;
}

/**
 * Computes gain that would be obtained if we would move vertex to community.
 * @param vertex           vertex number
 * @param prime            prime number used for hashing (and size of vertex's area in hash arrays)
 * @param community      neighbour's community
 * @param currentCommunity current community of vertex
 * @param communityWeight  community -> weight (sum of edges adjacent to vertices of community)
 * @param vertexEdgesSum   vertex -> sum of edges adjacent to vertex
 * @param hashCommunity    table informing which community's info is stored in given index
 * @param hashWeight       table of sum of weights between vertices and communities
 * @param hashTablesOffset offset of the vertex in hash arrays (single hash array may contain multiple vertices
 * @return gain that would be obtained by moving vertex to community
 */
__device__ float computeGain(const int    vertex, 
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

/**
 * Finds new vertex -> community assignment (stored in newVertexCommunity) that maximise gains for each vertex.
 * @param V                number of vertices
 * @param vertices         vertices
 * @param prime            prime number used for hashing
 * @param deviceStructures structures kept in device memory
 */

// grid dimension   = blocksNum X 1
// block dimension  = {4, 32} / {8, 16} / ...
// verticesBlock = blockDim.y
// hashWeight -> Though this is a pointer, it means a scalar
// hashWeight = hashCommunity[verticesBlock * prime]

// hashWeight       table of sum of weights between vertices and communities
// hashCommunity    table informing which community's info is stored in given index

// vertexToCommunity -> Though this is a pointer, it means a scalar
// vertexToCommunity = hashWeight[verticesBlock * prime]

// hashCommunity size = blockDimension.y * (prime * (intsize + floatsize) + floatsize)
__device__ void computeMove(const int    V, 
                            const int*   vertices, 
                            const int    prime, 
                                  device_structures deviceStructures, 
                                  int*   hashCommunity,
                                  float* hashWeight, 
                                  float* vertexToCurrentCommunity, 
                                  float* bestGains, 
                                  int*   bestCommunities) 
{
  int verticesPerBlock = blockDim.y;
  int vertexIndex      = blockIdx.x * verticesPerBlock + threadIdx.y;

  if(vertexIndex < V) 
  {
    int* vertexCommunity    = deviceStructures.vertexCommunity;
    int* edgesIndex         = deviceStructures.edgesIndex;
    int* edges              = deviceStructures.edges;
    int* communitySize      = deviceStructures.communitySize;
    int* newVertexCommunity = deviceStructures.newVertexCommunity;
    float* weights          = deviceStructures.weights;
    float* communityWeight  = deviceStructures.communityWeight;
    float* vertexEdgesSum   = deviceStructures.vertexEdgesSum;

    int concurrentNeighbours = blockDim.x;
    int hashTablesOffset     = threadIdx.y * prime;
    // hashTablesOffset = (0 ~ numVerticesPerBlock - 1) * prime

    if(threadIdx.x == 0)
      vertexToCurrentCommunity[threadIdx.y] = 0;

    for(unsigned int i = threadIdx.x; i < prime; i += concurrentNeighbours) 
    {
      hashWeight[hashTablesOffset + i] = 0;
      hashCommunity[hashTablesOffset + i] = -1;
    }

    if(concurrentNeighbours > WARP_SIZE)
      __syncthreads();

    int vertex           = vertices[vertexIndex];
    int currentCommunity = vertexCommunity[vertex];
    int bestCommunity    = currentCommunity;
    float bestGain = 0;
    // putting data in hash table
    int neighbourIndex = threadIdx.x + edgesIndex[vertex];
    int upperBound = edgesIndex[vertex + 1]; // ID of the last incident edge
    int curPos;

    // Visit all incident edges in the parallel manner
    // concurrent thread = blockdim.x
    while(neighbourIndex < upperBound) // This is ok because the edge IDs are in the sorted order
    {
      int neighbour = edges[neighbourIndex];
      int community = vertexCommunity[neighbour];
      float weight  = weights[neighbourIndex];
      // This lets us achieve ei -> C(i)\{i} instead of ei -> C(i)
      if(neighbour != vertex) 
      {
        curPos = prepareHashArrays(community, prime, weight, hashWeight, hashCommunity, hashTablesOffset);
        if(community == currentCommunity)
          atomicAdd(&vertexToCurrentCommunity[threadIdx.y], weight);
      }
      if( (community < currentCommunity || communitySize[community] > 1 || communitySize[currentCommunity] > 1) 
          && community != currentCommunity) 
      {
        // computeGain does not change anything
        // Gain is the cost of moving community i to community j
        float gain = computeGain(vertex, 
                                 community,         // community j
                                 currentCommunity,  // community i
                                 communityWeight, 
                                 vertexEdgesSum, 
                                 hashWeight[curPos]);

        if(gain > bestGain || (gain == bestGain && community < bestCommunity) ) 
        {
          bestGain      = gain;
          bestCommunity = community;
        }
      }
      neighbourIndex += concurrentNeighbours;
    }

    // WARP_SIZE := 32
    if(concurrentNeighbours <= WARP_SIZE) 
    {
      for(unsigned int offset = concurrentNeighbours / 2; offset > 0; offset /= 2) 
      {
        float otherGain      = __shfl_down_sync(FULL_MASK, bestGain, offset);
        int   otherCommunity = __shfl_down_sync(FULL_MASK, bestCommunity, offset);

        if(otherGain > bestGain || (otherGain == bestGain && otherCommunity < bestCommunity)) 
        {
          bestGain      = otherGain;
          bestCommunity = otherCommunity;
        }
      }
    } 
    else 
    {
      bestGains[threadIdx.x] = bestGain;
      bestCommunities[threadIdx.x] = bestCommunity;

      for(unsigned int offset = concurrentNeighbours / 2; offset > 0; offset /= 2) 
      {
        __syncthreads();

        if(threadIdx.x < offset) 
        {
          float otherGain = bestGains[threadIdx.x + offset];
          int otherCommunity = bestCommunities[threadIdx.x + offset];
          if(otherGain > bestGains[threadIdx.x] ||
            (otherGain == bestGains[threadIdx.x] && otherCommunity < bestCommunities[threadIdx.x])) 
          {
            bestGains[threadIdx.x] = otherGain;
            bestCommunities[threadIdx.x] = otherCommunity;
          }
        }
      }
      bestGain      = bestGains[threadIdx.x];
      bestCommunity = bestCommunities[threadIdx.x];
    }

    if(threadIdx.x == 0 && bestGain - vertexToCurrentCommunity[threadIdx.y] / M > 0) 
      newVertexCommunity[vertex] = bestCommunity;
    else 
      newVertexCommunity[vertex] = currentCommunity;
  }
}

                               // V == size of the int* vertices (== int* partition)
__global__ void computeMoveShared(const int  V, 
                                  const int* vertices, 
                                  const int  prime, 
                                  device_structures deviceStructures)
{
  int verticesPerBlock = blockDim.y;
  int vertexIndex      = blockIdx.x * verticesPerBlock + threadIdx.y;

  if(vertexIndex < V) 
  {
    extern __shared__ int s[];
    int*   hashCommunity            = s;
    auto*  hashWeight               = (float*) &hashCommunity[verticesPerBlock * prime];
    auto*  vertexToCurrentCommunity = (float*) &hashWeight[verticesPerBlock * prime];
    float* bestGains                = &vertexToCurrentCommunity[verticesPerBlock];
    int*   bestCommunities          = (int*) &bestGains[THREADS_PER_BLOCK];

		computeMove(V, 
			          vertices, 
			          prime, 
			          deviceStructures, 
			          hashCommunity, 
			          hashWeight, 
			          vertexToCurrentCommunity,
				        bestGains, 
				        bestCommunities);
  }
}

__global__ void computeMoveGlobal(const int    V, 
                                  const int*   vertices, 
                                  const int    prime, 
                                  device_structures deviceStructures, 
                                  int*   hashCommunity, 
                                  float* hashWeight)
{
  int verticesPerBlock = blockDim.y;
  int vertexIndex      = blockIdx.x * verticesPerBlock + threadIdx.y;

  if(vertexIndex < V) 
  {
    extern __shared__ int s[];
    auto *vertexToCurrentCommunity = (float *) s;
    float* bestGains = &vertexToCurrentCommunity[verticesPerBlock];
    int* bestCommunities = (int *) &bestGains[THREADS_PER_BLOCK];

    hashCommunity = hashCommunity + blockIdx.x * prime;
    hashWeight    = hashWeight    + blockIdx.x * prime;

	  computeMove(V, 
	              vertices, 
	              prime, 
	              deviceStructures, 
	              hashCommunity, 
	              hashWeight, 
	              vertexToCurrentCommunity,
	              bestGains, 
	              bestCommunities);
  }
}

/**
 * Updates vertexCommunity content based on newVertexCommunity content..
 * Additionally, updates communitySize.
 * @param V                number of vertices
 * @param vertices         vertices
 * @param deviceStructures structures kept in device memory
 */
__global__ void updateVertexCommunity(int V, 
                                      const int* vertices, 
                                      device_structures deviceStructures) 
{
  int index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

  if(index < V) 
  {
    int vertex = vertices[index];
    int oldCommunity = deviceStructures.vertexCommunity[vertex];
    int newCommunity = deviceStructures.newVertexCommunity[vertex];

    if(oldCommunity != newCommunity) 
    {
      deviceStructures.vertexCommunity[vertex] = newCommunity;
      atomicSub(&deviceStructures.communitySize[oldCommunity], 1);
      atomicAdd(&deviceStructures.communitySize[newCommunity], 1);
    }
  }
}

__global__ void updateOriginalToCommunity(device_structures deviceStructures) 
{
  int vertex = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

  if(vertex < *deviceStructures.originalV) 
  {
    int community = deviceStructures.originalToCommunity[vertex];
    deviceStructures.originalToCommunity[vertex] = deviceStructures.vertexCommunity[community];
    // VertexCommunity := Vertex -> Community
  }
}

struct isInBucket
{
  isInBucket(int llowerBound, int uupperBound, int* eedgesIndex) 
  {
    lowerBound = llowerBound;
    upperBound = uupperBound;
    edgesIndex = eedgesIndex;
  }

  int lowerBound, upperBound;
  int *edgesIndex;
  __host__ __device__
  bool operator()(const int &v) const
  {
    int edgesNumber = edgesIndex[v + 1] - edgesIndex[v];
    return edgesNumber > lowerBound && edgesNumber <= upperBound;
  }
};

int getMaxDegree(host_structures& hostStructures) 
{
  int curMax = 0;
  for(int i = 0; i < hostStructures.V; i++)
    curMax = std::max(curMax, hostStructures.edgesIndex[i+1] - hostStructures.edgesIndex[i]);
  return curMax;
}

bool optimiseModularity(float minGain, 
                        device_structures& deviceStructures, 
                          host_structures& hostStructures) 
{
  int V = hostStructures.V;
  // blocksNumber(int V, int numThreadPerVertex)
  // ~= (V * numThreadPerVertex) / 128
  computeEdgesSum<<<blocksNumber(V, WARP_SIZE), dim3{WARP_SIZE, THREADS_PER_BLOCK / WARP_SIZE}>>>(deviceStructures);
  HANDLE_ERROR(cudaMemcpy(hostStructures.edgesIndex, 
                        deviceStructures.edgesIndex,
                            (V + 1) * (sizeof(int)), 
                            cudaMemcpyDeviceToHost) );

  int* partition = deviceStructures.partition;
  thrust::sequence(thrust::device, partition, partition + V, 0);

  int    lastBucketPrime      = getPrime(getMaxDegree(hostStructures) * 1.5);
  int*   hashCommunity;
  float* hashWeight;
  int    lastBucketNum        = bucketsSize - 2;
  dim3   lastBlockDimension   = dims[lastBucketNum];
  // buckest[] = 0 4 8 16 32 84 319 INT_MAX
  auto   predicate            = isInBucket(buckets[lastBucketNum], buckets[lastBucketNum + 1], hostStructures.edgesIndex);
  int*   deviceVerticesEnd    = thrust::partition(thrust::device, partition, partition + V, predicate);
  int    verticesInLastBucket = thrust::distance(partition, deviceVerticesEnd);

  if(verticesInLastBucket > 0) 
  {
    unsigned int blocksNum = (verticesInLastBucket + lastBlockDimension.y - 1) / lastBlockDimension.y;
    HANDLE_ERROR(cudaMalloc((void**)&hashCommunity, lastBucketPrime * blocksNum * sizeof(int)   ));
    HANDLE_ERROR(cudaMalloc((void**)&hashWeight   , lastBucketPrime * blocksNum * sizeof(float) ));
  }

  float totalGain = minGain;
  bool wasAnythingChanged = false;

  while(totalGain >= minGain) 
  {
    float modularityBefore = calculateModularity(V, hostStructures.M, deviceStructures);

    // Algorithm 1) Line 4
    // bucketsSize = 8
    // 0th Iter : (0,    4] blockDimension = {  4, 32} prime =   7
    // 1st Iter : (5,    8] blockDimension = {  8, 16} prime =  13
    // 2nd Iter : (9,   16] blockDimension = { 16,  8} prime =  29
    // 3rd Iter : (17,  32] blockDimension = { 32,  4} prime =  53
    // 4th Iter : (33,  84] blockDimension = { 32,  4} prime = 127
    // 5th Iter : (85, 319] blockDimension = {128,  1} prime = 479
    for(int bucketNum = 0; bucketNum < bucketsSize - 2; bucketNum++)
    {
      dim3 blockDimension  = dims[bucketNum];
      int  prime           = primes[bucketNum]; 
      // primes = 7 13 29 53 127 479
      auto predicate       = isInBucket(buckets[bucketNum    ], 
                                        buckets[bucketNum + 1], 
                                        hostStructures.edgesIndex);
      deviceVerticesEnd    = thrust::partition(thrust::device, partition, partition + V, predicate);
      int verticesInBucket = thrust::distance(partition, deviceVerticesEnd);
      // Now, int* partition only has the index of the vertex whose degree is less than 4 / 8 / 16 / 32 / ~ 

      // if partition[] is not empty
      if(verticesInBucket > 0) 
      {
        int sharedMemSize =
          blockDimension.y * prime * (sizeof(float) + sizeof(int)) 
        + blockDimension.y * sizeof(float);

        if(blockDimension.x > WARP_SIZE)
          sharedMemSize += THREADS_PER_BLOCK * (sizeof(int) + sizeof(float));

        int blocksNum = (verticesInBucket + blockDimension.y - 1) / blockDimension.y;
        //  blocksNum = (size of int* partition) / 4 ( or 32, 16, 8 ...)

        computeMoveShared<<<blocksNum, blockDimension, sharedMemSize>>>(verticesInBucket, 
                                                                        partition, 
                                                                        prime,
                                                                        deviceStructures);
        // Update Vertex to Community Assignment
        updateVertexCommunity<<<blocksNumber(V, 1), THREADS_PER_BLOCK>>>(verticesInBucket, 
                                                                         partition,
                                                                         deviceStructures);

        // Update Community Weight
        thrust::fill(thrust::device, 
                     deviceStructures.communityWeight,
                     deviceStructures.communityWeight + hostStructures.V, 
                     0.0);

        computeCommunityWeight<<<blocksNumber(V, 1), THREADS_PER_BLOCK>>>(deviceStructures);
      }
    }

    // last bucket case
    // when vertex degree is larger than 319
    deviceVerticesEnd = thrust::partition(thrust::device, 
                                          partition, 
                                          partition + V, 
                                          predicate);

    int verticesInBucket = thrust::distance(partition, deviceVerticesEnd);

    if(verticesInBucket > 0) 
    {
      unsigned int blocksNum = (verticesInBucket + lastBlockDimension.y - 1) 
                             / lastBlockDimension.y;

      int sharedMemSize = THREADS_PER_BLOCK * (sizeof(int) + sizeof(float)) 
                        + lastBlockDimension.y * sizeof(float);

      computeMoveGlobal<<<blocksNum, lastBlockDimension, sharedMemSize>>>(verticesInBucket, 
                                                                          partition, 
                                                                          lastBucketPrime,
                                                                          deviceStructures, 
                                                                          hashCommunity, 
                                                                          hashWeight);
    }

    // updating vertex -> community assignment
    updateVertexCommunity<<<blocksNumber(V, 1), THREADS_PER_BLOCK>>>(verticesInBucket, 
                                                                     partition,
                                                                     deviceStructures);

    // updating community weight
    thrust::fill(thrust::device, 
                 deviceStructures.communityWeight,
                 deviceStructures.communityWeight + hostStructures.V, 
                 0.0);

    computeCommunityWeight<<<blocksNumber(V, 1), THREADS_PER_BLOCK>>>(deviceStructures);

    float modularityAfter = calculateModularity(V, hostStructures.M, deviceStructures);

    totalGain = modularityAfter - modularityBefore;

    wasAnythingChanged = wasAnythingChanged || (totalGain > 0);
  }

  HANDLE_ERROR(cudaMemcpy(hostStructures.vertexCommunity, 
               deviceStructures.vertexCommunity,
               hostStructures.V * sizeof(float), 
               cudaMemcpyDeviceToHost));

  if(verticesInLastBucket) 
  {
    HANDLE_ERROR(cudaFree(hashCommunity));
    HANDLE_ERROR(cudaFree(hashWeight));
  }

  updateOriginalToCommunity<<<blocksNumber(hostStructures.originalV, 1), THREADS_PER_BLOCK>>>(deviceStructures);
  return wasAnythingChanged;
}

__global__ void calculateToOwnCommunity(device_structures deviceStructures) 
{
  int verticesPerBlock = blockDim.y;
  int concurrentNeighbours = blockDim.x;
  float edgesSum = 0;
  int vertex = blockIdx.x * verticesPerBlock + threadIdx.y;
  int community = deviceStructures.vertexCommunity[vertex];

  if(vertex < *deviceStructures.V) 
  {
    int startOffset = deviceStructures.edgesIndex[vertex];
    int endOffset   = deviceStructures.edgesIndex[vertex + 1];

    for(int index = startOffset + threadIdx.x; index < endOffset; index += concurrentNeighbours) 
    {
      int neighbour = deviceStructures.edges[index];
      if(deviceStructures.vertexCommunity[neighbour] == community)
        edgesSum += deviceStructures.weights[index];
    }

    for(unsigned int offset = concurrentNeighbours / 2; offset > 0; offset /= 2) 
      edgesSum += __shfl_down_sync(FULL_MASK, edgesSum, offset);

    if(threadIdx.x == 0) 
      deviceStructures.toOwnCommunity[vertex] = edgesSum;
  }
}

float calculateModularity(int   V, 
                          float M, 
                          device_structures deviceStructures) 
{
  calculateToOwnCommunity<<<blocksNumber(V, WARP_SIZE), dim3{WARP_SIZE, THREADS_PER_BLOCK / WARP_SIZE}>>>(deviceStructures);

  // communityWeight of c := a_c (See Paper)
  float communityWeightSum 
    = thrust::transform_reduce(thrust::device, 
                               deviceStructures.communityWeight,
                               deviceStructures.communityWeight + V, 
                               square(), 
                               0.0, 
                               thrust::plus<float>());

  float toOwnCommunity 
    = thrust::reduce(thrust::device, 
                     deviceStructures.toOwnCommunity, 
                     deviceStructures.toOwnCommunity + V);

  return toOwnCommunity / (2 * M) - communityWeightSum  / (4 * M * M);
}

void copyDS2HS(device_structures& deviceStructures, 
                 host_structures& hostStructures) 
{
  std::vector<int> communityToVertexVector[hostStructures.V];
  
  HANDLE_ERROR(
      cudaMemcpy(hostStructures.originalToCommunity, 
                 deviceStructures.originalToCommunity,
                 hostStructures.originalV * sizeof(int), cudaMemcpyDeviceToHost) );

  for(int vertex = 0; vertex < hostStructures.originalV; vertex++) 
  {
    int community = hostStructures.originalToCommunity[vertex];
    communityToVertexVector[community].emplace_back(vertex);
  }
}

void initM(host_structures& hostStructures) 
{
  HANDLE_ERROR(cudaMemcpyToSymbol(M, &hostStructures.M, sizeof(float)));
}

};
