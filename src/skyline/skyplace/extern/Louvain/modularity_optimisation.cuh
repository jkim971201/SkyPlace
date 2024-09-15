#ifndef __MODULARITY_OPTIMISATION__CUH__
#define __MODULARITY_OPTIMISATION__CUH__
#include "utils.cuh"

namespace Louvain
{

// If we do not use 'static' keyword,
// then this will be visible in the my project
// and will make 'multiple definition' error.
__constant__ static float M;

const int bucketsSize = 8;
const int buckets[] = {0, 4, 8, 16, 32, 84, 319, INT_MAX};
const int primes[] = {7, 13, 29, 53, 127, 479};
// x - number of neighbours processed concurrently, y - vertices per block
const dim3 dims[] {
		{4, 32},
		{8, 16},
		{16, 8},
		{32, 4},
		{32, 4},
		{128, 1},
		{128, 1},
};

struct square 
{
  __device__ float operator()(const float &x) const 
  {
    return x * x;
  }
};

int getMaxDegree(host_structures& hostStructures);

/**
 * Function responsible for executing 1 phase (modularity optimisation)
 * @param minGain          minimum gain for going to next iteration of this phase
 * @param deviceStructures structures kept in device memory
 * @param hostStructures   structures kept in host memory
 * @return information whether any changes were applied
 */
bool optimiseModularity(float minGain, 
		               device_structures& deviceStructures, 
									   host_structures& hostStructures);

float calculateModularity(int V, float M, device_structures deviceStructures);

void copyDS2HS(device_structures& deviceStructures, host_structures& hostStructures);

void initM(host_structures& hostStructures);

};

#endif /* __MODULARITY_OPTIMISATION__CUH__ */
