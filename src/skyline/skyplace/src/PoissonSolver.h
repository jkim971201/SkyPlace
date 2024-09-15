#ifndef POISSON_SOLVER_H
#define POISSON_SOLVER_H

#include <memory>
#include <cstdio>
#include <cufft.h>

namespace skyplace 
{

class PoissonSolver
{
  public:
    PoissonSolver();
    PoissonSolver(int numBinX, int numBinY);

    ~PoissonSolver();

    void solvePoisson(const float* binDensity, 
                            float* potential,
                            float* electroForceX,
                            float* electroForceY);

    // Compute Potential Only (not Electric Force)
    void solvePoissonPotential(const float* binDensity,
                                     float* potential);

    void solvePoissonForce(float* electroForceX, 
                           float* electroForceY);

  private:

    int numBinX_;
    int numBinY_;

    void init();

    void setupForCUDAKernel();
    void freeDeviceMemory();

    cufftHandle plan_;
    cufftHandle planInverse_;

    cufftComplex* d_expkN_;
    cufftComplex* d_expkM_;

    cufftComplex* d_expkNForInverse_;
    cufftComplex* d_expkMForInverse_;

    cufftComplex* d_expkMN1_;
    cufftComplex* d_expkMN2_;

    cufftReal*    d_binDensity_;
    cufftReal*    d_auv_;
    cufftReal*    d_potential_;

    cufftReal*    d_efX_;
    cufftReal*    d_efY_;

    cufftReal*    d_workSpaceReal1_;
    cufftReal*    d_workSpaceReal2_;
    cufftReal*    d_workSpaceReal3_;

    cufftComplex* d_workSpaceComplex_;

    cufftReal*    d_inputForX_;
    cufftReal*    d_inputForY_;
};

}; // namespace skyplace 

#endif
