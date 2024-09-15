#pragma once

#include "utils.cuh"

namespace Louvain
{

void removeSparseCluster(int V,
		                     int minNumVertex, 
		                     host_structures& hostStructures, 
		                   device_structures& deviceStructures);

};


