#include <stdio.h>
#include "gpulouvain.h"
#include "utils.cuh"
#include "modularity_optimisation.cuh"
#include "community_aggregation.cuh"
#include "removeSparse.cuh"

namespace Louvain
{

void gpu_louvain(host_structures& hostStructures, float minGain, int minClusterSize)
{
  device_structures deviceStructures;
  aggregation_phase_structures aggregationPhaseStructures;

  copyStructures(hostStructures, deviceStructures, aggregationPhaseStructures);
  initM(hostStructures);

  int i = 0;
  int maxIter = 500;
  for(;;) 
  {
    bool worthContinue = optimiseModularity(minGain, deviceStructures, hostStructures);

		if(!worthContinue)
			break;

    aggregateCommunities(deviceStructures, hostStructures, aggregationPhaseStructures);

    int V;
    HANDLE_ERROR(cudaMemcpy(&V, deviceStructures.V, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Louvain Iter[%02d] V: %d Modularity: %f\n", i++, V, calculateModularity(V, hostStructures.M, deviceStructures));

    if(i == maxIter)
      break;
  }

  int V;
  HANDLE_ERROR(cudaMemcpy(&V, deviceStructures.V, sizeof(int), cudaMemcpyDeviceToHost));

//	int numSparseCluster = detectSparseCluster(V, minClusterSize, hostStructures, deviceStructures);
//  printf("Number of Sparse Cluster %d / %d \n", numSparseCluster, V);
//
//	if(numSparseCluster > 0)
//	{
//		printf("Modularity before removal: %f\n", calculateModularity(V, hostStructures.M, deviceStructures));
//    bool worthContinue = removeSparseCluster(minGain, minClusterSize, deviceStructures, hostStructures);
//    aggregateCommunities(deviceStructures, hostStructures, aggregationPhaseStructures);
//	}
//
//  HANDLE_ERROR(cudaMemcpy(&V, deviceStructures.V, sizeof(int), cudaMemcpyDeviceToHost));
//
//	numSparseCluster = detectSparseCluster(V, minClusterSize, hostStructures, deviceStructures);
//  printf("Number of Sparse Cluster %d / %d \n", numSparseCluster, V);

  copyDS2HS(deviceStructures, hostStructures);

  deleteStructures(deviceStructures, aggregationPhaseStructures);
}

};
