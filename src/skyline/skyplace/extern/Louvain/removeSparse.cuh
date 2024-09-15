#pragma once
#include "utils.cuh"
#include "modularity_optimisation.cuh"

namespace Louvain
{

int detectSparseCluster(int V,
                        int minNumVertex, 
                        host_structures& hostStructures, 
                      device_structures& deviceStructures);

bool removeSparseCluster(float minGain, 
		                       int minSize,
                        device_structures& deviceStructures, 
                          host_structures& hostStructures);
};
