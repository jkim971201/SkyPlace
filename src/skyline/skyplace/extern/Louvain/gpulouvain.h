#pragma once

#include "utils.cuh"
#include "modularity_optimisation.cuh"
#include "community_aggregation.cuh"

namespace Louvain
{

struct host_structures;

void gpu_louvain(host_structures& hostStructures, 
		             float minGain, 
								 int minClusterSize = 3);

};
