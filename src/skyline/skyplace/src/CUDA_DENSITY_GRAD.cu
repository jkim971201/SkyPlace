#include <cmath>
#include <vector>
#include <chrono>
#include <random>    // For mt19937
#include <algorithm> // For sort
#include <cstdio>
#include <fstream>

#include <cufft.h>

#include "CUDA_UTIL.h"
#include "DensityGradient.h"

namespace skyplace
{

__device__ inline float computeGaussian(float sigma,
                                        float binLx, 
                                        float binLy,
                                        float binUx,
                                        float binUy,
                                        float cx,
                                        float cy,
                                        float cellDx,
                                        float cellDy)
{
  // Assume that mean_x = 0.0, mean_y = 0.0
  float x = ( (binLx + binUx) / 2.0 - cx ) / (2.0 * cellDx);
  float y = ( (binLy + binUy) / 2.0 - cy ) / (2.0 * cellDy);

  float gaussianX = __expf( -0.5 * (x / sigma) * (x / sigma) );
  float gaussianY = __expf( -0.5 * (y / sigma) * (y / sigma) );

	// wmax = 1.0?
  return gaussianX * gaussianY / ( 2 * float(FFT_PI) * sigma * sigma);
}

__device__ inline OVBIN findBinWithDensitySizeDevice(const int   numBinX,
                                                     const int   numBinY,
                                                     const float binWidth,
                                                     const float binHeight,
                                                     const float dieLx,
                                                     const float dieLy,
                                                     const float dieUx,
                                                     const float dieUy,
                                                     const float cellCx,
                                                     const float cellCy,
                                                     const float cellDx,
                                                     const float cellDy)
{
  OVBIN ovBins;

  /// X
  float lx = cellCx - cellDx/2;
  float ux = cellCx + cellDx/2;
  
  int minX = floor((lx - dieLx) / binWidth);
  int maxX = ceil ((ux - dieLx) / binWidth);

  minX = max(minX, 0);
  maxX = min(numBinX, maxX);

  ovBins.lxID = minX;
  ovBins.uxID = maxX;

  /// Y
  float ly = cellCy - cellDy / 2;
  float uy = cellCy + cellDy / 2;

  int minY = floor((ly - dieLy) / binHeight);
  int maxY = ceil ((uy - dieLy) / binHeight);

  minY = max(minY, 0);
  maxY = min(numBinY, maxY);

  ovBins.lyID = minY;
  ovBins.uyID = maxY;

  return ovBins;
}

__global__ void initCellAreaOfEachBin(const int totalNumBin,
                                      float* movableArea,
                                      float* fillerArea)
{
  // i := binID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < totalNumBin)
  {
    movableArea[i] = 0;
    fillerArea[i] = 0;
  }
}

__global__ void computeDensityContributionOfEachCell(
    const int      numMovable,
    const int      numBinX,
    const int      numBinY,
    const float    binWidth,
    const float    binHeight,
    const float    targetDensity,
    const float*   binLx,
    const float*   binLy,
    const float*   binUx,
    const float*   binUy,
    const float    dieLx,
    const float    dieLy,
    const float    dieUx,
    const float    dieUy,
    const float*   cellCx,
    const float*   cellCy,
    const float*   cellDx,
    const float*   cellDy,
    const float*   cellDensityScale,
    const bool*    isFiller,
    const bool*    isMacro,
          float*   fillerArea,
          float*   movableArea,
          float    sigma)
{
  // i := cellID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numMovable)
  {
    float cellCxCoordi = cellCx[i];
    float cellCyCoordi = cellCy[i];

    float cellDWidth  = cellDx[i];
    float cellDHeight = cellDy[i];

    float celldLx = cellCxCoordi - cellDWidth / 2.0;
    float celldUx = cellCxCoordi + cellDWidth / 2.0;

    float celldLy = cellCyCoordi - cellDHeight / 2.0;
    float celldUy = cellCyCoordi + cellDHeight / 2.0;

    OVBIN ovBins 
      = findBinWithDensitySizeDevice(numBinX, 
                                     numBinY,
                                     binWidth,  
                                     binHeight, 
                                     dieLx, 
                                     dieLy, 
                                     dieUx, 
                                     dieUy, 
                                     cellCxCoordi,
                                     cellCyCoordi,
                                     cellDWidth,
                                     cellDHeight);

    auto getOverlapLength = [] (float cellMin, float cellMax, float binMin, float binMax)
    {
      return max(0.0, min(cellMax, binMax) - max(cellMin, binMin));
    };

    int lxID = ovBins.lxID;
    int uxID = ovBins.uxID;

    int lyID = ovBins.lyID;
    int uyID = ovBins.uyID;

    for(int j = lxID; j < uxID; j++)
    {
      for(int k = lyID; k < uyID; k++)
      {
        int binID = j + k * numBinX;

        float binLxCoordi = binLx[binID];
        float binLyCoordi = binLy[binID];
        float binUxCoordi = binUx[binID];
        float binUyCoordi = binUy[binID];

        float rectWidth = getOverlapLength(celldLx, 
                                           celldUx,
                                           binLxCoordi, 
                                           binUxCoordi);

        float rectHeight = getOverlapLength(celldLy, 
                                            celldUy, 
                                            binLyCoordi, 
                                            binUyCoordi);

        float overlapArea = 0.0;
        // Macro should be scaled-down with target-density
        // by OpenROAD RePlAce comments
        if(isMacro[i])
				{
          overlapArea = rectWidth * rectHeight * cellDensityScale[i] * targetDensity;
				}
        else
          overlapArea = rectWidth * rectHeight * cellDensityScale[i];

        if(isFiller[i])
          atomicAdd(&(fillerArea[binID]), overlapArea);
        else
          atomicAdd(&(movableArea[binID]), overlapArea);
      }
    }
  }
}

__global__ void computeOverflow(const int    totalNumBin, 
                                const float* fixedArea,
                                const float* fillerArea,
                                const float* movableArea,
                                const float* scaledBinArea,
                                      float* binDensity,
                                      float* overflowArea)
{
  const unsigned int binID = blockIdx.x * blockDim.x + threadIdx.x;

  if(binID < totalNumBin)
  {
    float binArea = scaledBinArea[binID];

    binDensity[binID] = ( (movableArea[binID]) 
                        + (  fixedArea[binID])
                        + ( fillerArea[binID]) ) / binArea;

    overflowArea[binID] = max(0.0, (movableArea[binID]) 
                                 + (  fixedArea[binID]) - binArea);
  }
}

__global__ void computeMacroOverflow(const int    totalNumBin, 
                                     const float* macroArea,
                                     const float* scaledBinArea,
                                            float* macroOverflow)
{
  const unsigned int binID = blockIdx.x * blockDim.x + threadIdx.x;

  if(binID < totalNumBin)
  {
    float binArea = scaledBinArea[binID];
    macroOverflow[binID] = max(0.0, macroArea[binID] - binArea);
  }
}

__global__ void getDensityGradForEachCell(const int    numMovable, 
                                          const int    numBinX,
                                          const int    numBinY,
                                          const float  binWidth,
                                          const float  binHeight,
                                          const float* binLx,
                                          const float* binLy,
                                          const float* binUx,
                                          const float* binUy,
                                          const float  dieLx,
                                          const float  dieLy,
                                          const float  dieUx,
                                          const float  dieUy,
                                          const float* cellCx,
                                          const float* cellCy,
                                          const float* cellDx,
                                          const float* cellDy,
                                          const float* cellDensityScale,
                                          const float* binLambda,
                                          const float* electroForceX,
                                          const float* electroForceY,
                                                float* densityGradX,
                                                float* densityGradY,
                                                float* densityPreconditioner)
{
  // i := cellID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numMovable)
  {
    OVBIN ovBins 
      = findBinWithDensitySizeDevice(numBinX, 
                                     numBinY,
                                     binWidth,
                                     binHeight, 
                                     dieLx, 
                                     dieLy, 
                                     dieUx, 
                                     dieUy, 
                                     cellCx[i],
                                     cellCy[i],
                                     cellDx[i],
                                     cellDy[i]);

    auto getOverlapLength = [] (float cellMin, float cellMax, float binMin, float binMax)
    {
      return max(0.0, min(cellMax, binMax) - max(cellMin, binMin));
    };

    float electroForceSumX = 0;
    float electroForceSumY = 0;

    float preconditioner = 0.0;

    for(int j = ovBins.lxID; j < ovBins.uxID; j++)
    {
      for(int k = ovBins.lyID; k < ovBins.uyID; k++)
      {
        int binID = j + k * numBinX;

        // Local Lambda
        float lambda = binLambda[binID];

        float celldLx = cellCx[i] - cellDx[i] / 2.0;
        float celldUx = cellCx[i] + cellDx[i] / 2.0;

        float rectWidth = getOverlapLength(celldLx, 
                                           celldUx,
                                           binLx[binID], 
                                           binUx[binID]);

        float celldLy = cellCy[i] - cellDy[i] / 2.0;
        float celldUy = cellCy[i] + cellDy[i] / 2.0;

        float rectHeight = getOverlapLength(celldLy, 
                                            celldUy, 
                                            binLy[binID], 
                                            binUy[binID]);

        float overlapArea = rectWidth * rectHeight * cellDensityScale[i];

        preconditioner += lambda * overlapArea;

        // GPU version of Poisson Solver is
        // multiplied by sqrt(2) * sqrt(2) 
        // Therefore, electroForce is multiplied by 0.5
        electroForceSumX += lambda * overlapArea * electroForceX[binID] * 0.5;
        electroForceSumY += lambda * overlapArea * electroForceY[binID] * 0.5;
      }
    }
    densityGradX[i] = electroForceSumX;
    densityGradY[i] = electroForceSumY;
    densityPreconditioner[i] = preconditioner;
  }
}

__global__ void getPotentialDrivenDensityGrad(const int    numMovable, 
                                              const int    numBinX,
                                              const int    numBinY,
                                              const float  binWidth,
                                              const float  binHeight,
                                              const float* binLx,
                                              const float* binLy,
                                              const float* binUx,
                                              const float* binUy,
                                              const float  dieLx,
                                              const float  dieLy,
                                              const float  dieUx,
                                              const float  dieUy,
                                              const float* cellCx,
                                              const float* cellCy,
                                              const float* cellDx,
                                              const float* cellDy,
                                              const float* cellDensityScale,
                                              const float* binLambda,
                                              const float* binPotential,
                                                    float* densityGradX,
                                                    float* densityGradY,
                                                    float* densityPreconditioner)
{
  // i := cellID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numMovable)
  {
    OVBIN ovBins 
      = findBinWithDensitySizeDevice(numBinX, 
                                     numBinY,
                                     binWidth,
                                     binHeight, 
                                     dieLx, 
                                     dieLy, 
                                     dieUx, 
                                     dieUy, 
                                     cellCx[i],
                                     cellCy[i],
                                     cellDx[i],
                                     cellDy[i]);

    auto getOverlapLength = [] (float cellMin, float cellMax, float binMin, float binMax)
    {
      return max(0.0, min(cellMax, binMax) - max(cellMin, binMin));
    };

    float electroForceSumX = 0;
    float electroForceSumY = 0;

    float preconditioner = 0.0;

    for(int k = ovBins.lyID; k < ovBins.uyID; k++)
    {
      for(int j = ovBins.lxID; j < ovBins.uxID; j++)
      {
        int binID = j + k * numBinX;

        // Local Lambda
        float lambda = binLambda[binID];

        float celldLx = cellCx[i] - cellDx[i] / 2.0;
        float celldUx = cellCx[i] + cellDx[i] / 2.0;

        float rectWidth = getOverlapLength(celldLx, 
                                           celldUx,
                                           binLx[binID], 
                                           binUx[binID]);

        float celldLy = cellCy[i] - cellDy[i] / 2.0;
        float celldUy = cellCy[i] + cellDy[i] / 2.0;

        float rectHeight = getOverlapLength(celldLy, 
                                            celldUy, 
                                            binLy[binID], 
                                            binUy[binID]);

        float overlapArea = rectWidth * rectHeight * cellDensityScale[i];

        preconditioner += lambda * overlapArea;

        electroForceSumX += lambda * rectHeight * binPotential[binID] 
                            * getPartialDerivative(celldLx, celldUx, binLx[binID], binUx[binID]);
        electroForceSumY += lambda * rectWidth  * binPotential[binID] 
                            * getPartialDerivative(celldLy, celldUy, binLy[binID], binUy[binID]);
      }
    }
    densityGradX[i] = -1.0 * electroForceSumX * cellDensityScale[i];
    densityGradY[i] = -1.0 * electroForceSumY * cellDensityScale[i];
    densityPreconditioner[i] = preconditioner;
  }
}

__global__ void updateLocalLambda(const int    totalNumBin, 
                                  const float  targetDensity,
                                  const float* binDensity,
                                        float* binLambda)
{
  const unsigned int binID = blockIdx.x * blockDim.x + threadIdx.x;

  if(binID < totalNumBin)
  {
    float density = binDensity[binID];

    if(density > targetDensity)
      binLambda[binID] = __powf(1 + __log10f(density), 0.5);
    else
      binLambda[binID] = 1.0;
  }
}

void
DensityGradient::computeGrad(float* densityGradX,
                             float* densityGradY,
                             const float* cellCx,
                             const float* cellCy)
{
  int numThreadBin = 256;
  int numBlockBin  = (numBinX_ * numBinY_ - 1 + numThreadBin) / numThreadBin;

  int numThreadCell = 256;
  int numBlockCell  = (numMovable_ * 2 - 1 + numThreadCell) / numThreadCell;

  auto t0 = std::chrono::high_resolution_clock::now();

  /////////////////
  // CUDA Kernel //
  // Step 1. Initialize movableArea as zero
  //initCellAreaOfEachBin<<<numBlockBin, numThreadBin>>>(numBinX_ * numBinY_, 
  //                                                       d_ptr_movableArea_,
  //                                                       d_ptr_fillerArea_);

  thrust::fill(d_movableArea_.begin(), d_movableArea_.end(), 0.0);
  thrust::fill(d_fillerArea_.begin() , d_fillerArea_.end() , 0.0);
  // thrust::fill(d_macroArea_.begin()  , d_macroArea_.end()  , 0.0);

  // Step 2. Compute OverlapArea
  computeDensityContributionOfEachCell<<<numBlockCell, numThreadCell>>>(numMovable_, 
                                                                        numBinX_, 
                                                                        numBinY_, 
                                                                        binWidth_, 
                                                                        binHeight_,
                                                                        targetDensity_,
                                                                        d_ptr_binLx_,
                                                                        d_ptr_binLy_,
                                                                        d_ptr_binUx_,
                                                                        d_ptr_binUy_,
                                                                        dieLx_, 
                                                                        dieLy_, 
                                                                        dieUx_, 
                                                                        dieUy_,
                                                                        cellCx,
                                                                        cellCy,
                                                                        d_ptr_cellDensityWidth_,
                                                                        d_ptr_cellDensityHeight_,
                                                                        d_ptr_cellDensityScale_,
                                                                        d_ptr_isFiller_,
                                                                        d_ptr_isMacro_,
                                                                        d_ptr_fillerArea_,
                                                                        d_ptr_movableArea_,
																																				sigma_);

  // Step 3. Update Overflow
  computeOverflow<<<numBlockBin, numThreadBin>>>(numBinX_ * numBinY_  ,
                                                 d_ptr_fixedArea_     ,
                                                 d_ptr_fillerArea_    ,
                                                 d_ptr_movableArea_   ,
                                                 d_ptr_scaledBinArea_ ,
                                                 d_ptr_binDensity_    ,
                                                 d_ptr_overflowArea_);

//  // Extra Step : Update Macro Overflow
//  computeMacroOverflow<<<numBlockBin, numThreadBin>>>(numBinX_ * numBinY_,
//                                                      d_ptr_macroArea_,
//                                                      d_ptr_scaledBinArea_,
//                                                      d_ptr_macroOverflowArea_);

  auto t1 = std::chrono::high_resolution_clock::now();

  overflow_ = thrust::reduce(d_overflowArea_.begin(),
                             d_overflowArea_.end()) / sumMovableArea_;

  macroOverflow_ = thrust::reduce(d_macroOverflowArea_.begin(), 
                                  d_macroOverflowArea_.end()) / sumMovableArea_;

  std::chrono::duration<double> runtime1 = t1 - t0;
  binDenUpTime_ += runtime1.count();

  // Step 4. Solve Poisson Equation
  poissonSolver_->solvePoisson(d_ptr_binDensity_, 
                               d_ptr_binPotential_, 
                               d_ptr_electroForceX_, 
                               d_ptr_electroForceY_);

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> runtime2 = t2 - t1;

  poissonTime_ += runtime2.count();

  // Step 5. Compute Density Gradient based on Electric Force XY
  getDensityGradForEachCell<<<numBlockCell, numThreadCell>>>(numMovable_, 
                                                             numBinX_,
                                                             numBinY_,
                                                             binWidth_, 
                                                             binHeight_,
                                                             d_ptr_binLx_,
                                                             d_ptr_binLy_,
                                                             d_ptr_binUx_,
                                                             d_ptr_binUy_,
                                                             dieLx_, 
                                                             dieLy_,
                                                             dieUx_, 
                                                             dieUy_,
                                                             cellCx,
                                                             cellCy,
                                                             d_ptr_cellDensityWidth_,
                                                             d_ptr_cellDensityHeight_,
                                                             d_ptr_cellDensityScale_,
                                                             d_ptr_binLambda_,
                                                             d_ptr_electroForceX_,
                                                             d_ptr_electroForceY_,
                                                             densityGradX,
                                                             densityGradY,
                                                             d_ptr_densityPreconditioner_);

//  // Step 5. Compute Density Gradient based on Electric Potential
//  getPotentialDrivenDensityGrad<<<numBlockCell, numThreadCell>>>(numMovable_, 
//                                                                 numBinX_,
//                                                                 numBinY_,
//                                                                 binWidth_, 
//                                                                 binHeight_,
//                                                                 d_ptr_binLx_,
//                                                                 d_ptr_binLy_,
//                                                                 d_ptr_binUx_,
//                                                                 d_ptr_binUy_,
//                                                                 dieLx_, 
//                                                                 dieLy_,
//                                                                 dieUx_, 
//                                                                 dieUy_,
//                                                                 cellCx,
//                                                                 cellCy,
//                                                                 d_ptr_cellDensityWidth_,
//                                                                 d_ptr_cellDensityHeight_,
//                                                                 d_ptr_cellDensityScale_,
//                                                                 d_ptr_binLambda_,
//                                                                 d_ptr_binPotential_,
//                                                                 densityGradX,
//                                                                 densityGradY,
//                                                                 d_ptr_densityPreconditioner_);


  if(localLambdaMode_)
  {
    //auto minMax  = thrust::minmax_element(thrust::device, 
    //                           d_binDensity_.begin(), 
    //                           d_binDensity_.end());

    //float minDen = *(minMax.first);
    //float maxDen = *(minMax.second);

    //printf("minDen = %f  maxDen = %f\n", minDen, maxDen);

    updateLocalLambda<<<numBlockBin, numThreadBin>>>(numBinX_ * numBinY_,
                                                     1.0,
                                                     d_ptr_binDensity_,
                                                     d_ptr_binLambda_);
  }

  auto t3 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> runtime3 = t3 - t0;

  densityTime_ += runtime3.count();
}

void
DensityGradient::computeGradAfterValue(float* densityGradX, 
                                       float* densityGradY,
                                       const float* cellCx, 
                                       const float* cellCy)
{
  int numThreadCell = 256;
  int numBlockCell  = (numMovable_ * 2 - 1 + numThreadCell) / numThreadCell;

  auto t1 = std::chrono::high_resolution_clock::now();

  // Step 4. Solve Poisson Equation
  poissonSolver_->solvePoissonForce(d_ptr_electroForceX_, 
                                    d_ptr_electroForceY_);

  // Step 5. Compute Density Gradient based on Electric Force XY
  getDensityGradForEachCell<<<numBlockCell, numThreadCell>>>(numMovable_, 
                                                             numBinX_,
                                                             numBinY_,
                                                             binWidth_, 
                                                             binHeight_,
                                                             d_ptr_binLx_,
                                                             d_ptr_binLy_,
                                                             d_ptr_binUx_,
                                                             d_ptr_binUy_,
                                                             dieLx_, 
                                                             dieLy_,
                                                             dieUx_, 
                                                             dieUy_,
                                                             cellCx,
                                                             cellCy,
                                                             d_ptr_cellDensityWidth_,
                                                             d_ptr_cellDensityHeight_,
                                                             d_ptr_cellDensityScale_,
                                                             d_ptr_binLambda_,
                                                             d_ptr_electroForceX_,
                                                             d_ptr_electroForceY_,
                                                             densityGradX,
                                                             densityGradY,
                                                             d_ptr_densityPreconditioner_);

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> runtime = t2 - t1;

  densityTime_ += runtime.count();
}

float 
DensityGradient::computePenalty(const float* d_ptr_cellCx, 
                                const float* d_ptr_cellCy)
{
  int numThreadBin = 256;
  int numBlockBin  = (numBinX_ * numBinY_ - 1 + numThreadBin) / numThreadBin;

  int numThreadCell = 256;
  int numBlockCell  = (numMovable_ * 2 - 1 + numThreadCell) / numThreadCell;

  /////////////////
  // CUDA Kernel //
  // Step 1. Initialize movableArea as zero
  initCellAreaOfEachBin<<<numBlockBin, numThreadBin>>>(numBinX_ * numBinY_, 
                                                       d_ptr_movableArea_,
                                                       d_ptr_fillerArea_);

  // Step 2. Compute OverlapArea
  computeDensityContributionOfEachCell<<<numBlockCell, numThreadCell>>>(numMovable_, 
                                                                        numBinX_, 
                                                                        numBinY_, 
                                                                        binWidth_, 
                                                                        binHeight_,
                                                                        targetDensity_,
                                                                        d_ptr_binLx_,
                                                                        d_ptr_binLy_,
                                                                        d_ptr_binUx_,
                                                                        d_ptr_binUy_,
                                                                        dieLx_, 
                                                                        dieLy_, 
                                                                        dieUx_, 
                                                                        dieUy_,
                                                                        d_ptr_cellCx,
                                                                        d_ptr_cellCy,
                                                                        d_ptr_cellDensityWidth_,
                                                                        d_ptr_cellDensityHeight_,
                                                                        d_ptr_cellDensityScale_,
                                                                        d_ptr_isFiller_,
                                                                        d_ptr_isMacro_,
                                                                        d_ptr_fillerArea_,
                                                                        d_ptr_movableArea_,
																																				sigma_);

  // Step 3. Update Overflow
  computeOverflow<<<numBlockBin, numThreadBin>>>(numBinX_ * numBinY_  ,
                                                 d_ptr_fixedArea_     ,
                                                 d_ptr_fillerArea_    ,
                                                 d_ptr_movableArea_   ,
                                                 d_ptr_scaledBinArea_ ,
                                                 d_ptr_binDensity_    ,
                                                 d_ptr_overflowArea_);

  // Step 4. Solve Poisson Equation (computePotentialOnly)
  poissonSolver_->solvePoissonPotential(d_ptr_binDensity_, 
                                        d_ptr_binPotential_);

  thrust::transform(d_binPotential_.begin(), 
                    d_binPotential_.end(), 
                    d_binDensity_.begin(), 
                    d_binPenalty_.begin(),
                    thrust::multiplies<float>());

  thrust::transform(d_binPenalty_.begin(), 
                    d_binPenalty_.end(), 
                    d_binLambda_.begin(), 
                    d_binPenalty_.begin(),
                    thrust::multiplies<float>());

  sumPenalty_  = thrust::reduce(d_binPenalty_.begin(), d_binPenalty_.end());
  sumPenalty_ *= (db_->binX() * db_->binY() * db_->binX() * db_->binY() );
  // multiply binArea

  return sumPenalty_;
}

float
DensityGradient::computePenaltyAfterGrad()
{
  thrust::transform(d_binPotential_.begin(), 
                    d_binPotential_.end(), 
                    d_binDensity_.begin(), 
                    d_binPenalty_.begin(),
                    thrust::multiplies<float>());

  thrust::transform(d_binPenalty_.begin(), 
                    d_binPenalty_.end(), 
                    d_binLambda_.begin(), 
                    d_binPenalty_.begin(),
                    thrust::multiplies<float>());

  sumPenalty_  = thrust::reduce(d_binPenalty_.begin(), d_binPenalty_.end());
  sumPenalty_ *= (db_->binX() * db_->binY() * 0.5);

  return sumPenalty_;
}

void
DensityGradient::updateStdDensityWeight(float coeffStd)
{
  auto transformOp = checkAndMultiply(coeffStd);

  thrust::transform(d_cellDensityScale_.begin(),
			              d_cellDensityScale_.end(),
                    d_isStd_.begin(),
                    d_cellDensityScale_.begin(),
                    transformOp );
}

void
DensityGradient::updateMacroDensityWeight(float coeffMacro)
{
  auto transformOp = checkAndMultiply(coeffMacro);

  thrust::transform(d_cellDensityScale_.begin(),
			              d_cellDensityScale_.end(),
                    d_isMacro_.begin(),
                    d_cellDensityScale_.begin(),
                    transformOp );
}

void
DensityGradient::updateFillerDensityWeight(float coeffFiller)
{
  auto transformOp = checkAndMultiply(coeffFiller);

  thrust::transform(d_cellDensityScale_.begin(),
			              d_cellDensityScale_.end(),
                    d_isFiller_.begin(),
                    d_cellDensityScale_.begin(),
                    transformOp);
}

void
DensityGradient::initForCUDAKernel()
{
  printf("[CUDA-DENSITY] Start Initialization.\n");
  
  int numBin2 = numBinX_ * numBinY_;

  thrust::host_vector<float> h_binLambda(numBin2);

  thrust::host_vector<float> h_fixedArea(numBin2);
  thrust::host_vector<float> h_scaledBinArea(numBin2);
  thrust::host_vector<float> h_binLx(numBin2);
  thrust::host_vector<float> h_binLy(numBin2);
  thrust::host_vector<float> h_binUx(numBin2);
  thrust::host_vector<float> h_binUy(numBin2);

  d_ptr_isStd_          = setThrustVector<bool>(numMovable_, d_isStd_);
  d_ptr_isFiller_       = setThrustVector<bool>(numMovable_, d_isFiller_);
  d_ptr_isMacro_        = setThrustVector<bool>(numMovable_, d_isMacro_);

  d_ptr_fixedArea_      = setThrustVector<float>(numBin2, d_fixedArea_);
  d_ptr_macroArea_      = setThrustVector<float>(numBin2, d_macroArea_);
  d_ptr_scaledBinArea_  = setThrustVector<float>(numBin2, d_scaledBinArea_);

  d_ptr_binLx_          = setThrustVector<float>(numBin2, d_binLx_);
  d_ptr_binLy_          = setThrustVector<float>(numBin2, d_binLy_);
  d_ptr_binUx_          = setThrustVector<float>(numBin2, d_binUx_);
  d_ptr_binUy_          = setThrustVector<float>(numBin2, d_binUy_);

  d_ptr_movableArea_       = setThrustVector<float>(numBin2, d_movableArea_);
  d_ptr_fillerArea_        = setThrustVector<float>(numBin2, d_fillerArea_);
  d_ptr_overflowArea_      = setThrustVector<float>(numBin2, d_overflowArea_);
  d_ptr_macroOverflowArea_ = setThrustVector<float>(numBin2, d_macroOverflowArea_);

  d_ptr_binDensity_     = setThrustVector<float>(numBin2, d_binDensity_);
  d_ptr_binLambda_      = setThrustVector<float>(numBin2, d_binLambda_);
  d_ptr_binPotential_   = setThrustVector<float>(numBin2, d_binPotential_);
  d_ptr_binPenalty_     = setThrustVector<float>(numBin2, d_binPenalty_);

  d_ptr_densityPreconditioner_ 
    = setThrustVector<float>(numMovable_, d_densityPreconditioner_);

  d_ptr_electroForceX_  = setThrustVector<float>(numBin2, d_electroForceX_);
  d_ptr_electroForceY_  = setThrustVector<float>(numBin2, d_electroForceY_);

  d_ptr_cellDensityWidth_  = setThrustVector<float>(numMovable_, d_cellDensityWidth_);
  d_ptr_cellDensityHeight_ = setThrustVector<float>(numMovable_, d_cellDensityHeight_);
  d_ptr_cellDensityScale_  = setThrustVector<float>(numMovable_, d_cellDensityScale_);

  int binIdx = 0;
  for(auto& bin : db_->bins())
  {
    h_fixedArea[binIdx]     
      = bin->fixedArea();
    h_scaledBinArea[binIdx] 
      = bin->area() * bin->targetDensity();

    h_binLx[binIdx] = bin->lx();
    h_binLy[binIdx] = bin->ly();
    h_binUx[binIdx] = bin->ux();
    h_binUy[binIdx] = bin->uy();

    h_binLambda[binIdx] = bin->lambda();

    binIdx++;
  }

  thrust::host_vector<bool>  h_isStd(numMovable_);
  thrust::host_vector<bool>  h_isFiller(numMovable_);
  thrust::host_vector<bool>  h_isMacro(numMovable_);
  thrust::host_vector<float> h_cellDensityWidth(numMovable_);
  thrust::host_vector<float> h_cellDensityHeight(numMovable_);
  thrust::host_vector<float> h_cellDensityScale(numMovable_);

  int i = 0;
  for(auto& cell : db_->movableCells())
  {
    h_isStd[i]             = (!cell->isFiller() && !cell->isMacro() && !cell->isFixed());
    h_isFiller[i]          = cell->isFiller();
    h_isMacro[i]           = cell->isMacro();
    h_cellDensityWidth[i]  = cell->dDx();
    h_cellDensityHeight[i] = cell->dDy();
    h_cellDensityScale[i]  = cell->densityScale();
    i++;
  }

  // Host -> Device
  thrust::copy(h_isStd.begin(),
               h_isStd.end(),
               d_isStd_.begin());

  thrust::copy(h_isFiller.begin(),
               h_isFiller.end(),
               d_isFiller_.begin());

  thrust::copy(h_isMacro.begin(),
               h_isMacro.end(),
               d_isMacro_.begin());

  thrust::copy(h_cellDensityWidth.begin(),
               h_cellDensityWidth.end(),
               d_cellDensityWidth_.begin());

  thrust::copy(h_cellDensityHeight.begin(),
               h_cellDensityHeight.end(),
               d_cellDensityHeight_.begin());

  thrust::copy(h_cellDensityScale.begin(),
               h_cellDensityScale.end(),
               d_cellDensityScale_.begin());

  thrust::copy(h_binLambda.begin(),
               h_binLambda.end(),
               d_binLambda_.begin());

  thrust::copy(h_fixedArea.begin(),
               h_fixedArea.end(),
               d_fixedArea_.begin());

  thrust::copy(h_scaledBinArea.begin(),
               h_scaledBinArea.end(),
               d_scaledBinArea_.begin());

  thrust::copy(h_binLx.begin(),
               h_binLx.end(),
               d_binLx_.begin());

  thrust::copy(h_binLy.begin(),
               h_binLy.end(),
               d_binLy_.begin());

  thrust::copy(h_binUx.begin(),
               h_binUx.end(),
               d_binUx_.begin());

  thrust::copy(h_binUy.begin(),
               h_binUy.end(),
               d_binUy_.begin());

  printf("[CUDA-DENSITY] Finish Initialization. \n");
}

DensityGradient::DensityGradient()
  :  
    db_                      (nullptr),
    numBinX_                 (0),
    numBinY_                 (0),
    numMovable_              (0),
    sumMovableArea_          (0),
    overflow_                (0),

    dieLx_                   (0),
    dieLy_                   (0),
    dieUx_                   (0),
    dieUy_                   (0),
    targetDensity_           (0),
    binWidth_                (0),
    binHeight_               (0),
		sigma_                   (1.0),
    localLambdaMode_         (false),

    d_ptr_isFiller_          (nullptr),
    d_ptr_isMacro_           (nullptr),
    d_ptr_macroArea_         (nullptr),
    d_ptr_fixedArea_         (nullptr),
    d_ptr_scaledBinArea_     (nullptr),
    d_ptr_binLx_             (nullptr),
    d_ptr_binLy_             (nullptr),
    d_ptr_binUx_             (nullptr),
    d_ptr_binUy_             (nullptr),
    d_ptr_movableArea_       (nullptr),
    d_ptr_fillerArea_        (nullptr),
    d_ptr_overflowArea_      (nullptr),
    d_ptr_macroOverflowArea_ (nullptr),
    d_ptr_binDensity_        (nullptr),
    d_ptr_binPotential_      (nullptr),
    d_ptr_binPenalty_        (nullptr),
    d_ptr_electroForceX_     (nullptr),
    d_ptr_electroForceY_     (nullptr),

    densityTime_             (0.0),
    poissonTime_             (0.0),
    binDenUpTime_            (0.0)
{}

DensityGradient::DensityGradient(std::shared_ptr<SkyPlaceDB> db)
  : DensityGradient()
{
  printf("[DensityGradient] Start Initialization.\n");

  db_ = db;

  numBinX_        = db_->numBinX();
  numBinY_        = db_->numBinY();
  numMovable_     = db_->numMovable();
  sumMovableArea_ = db_->sumMovableArea();

  dieLx_ = db_->die()->lx();
  dieLy_ = db_->die()->ly();
  dieUx_ = db_->die()->ux();
  dieUy_ = db_->die()->uy();

  binWidth_  = db_->binX();
  binHeight_ = db_->binY();

  targetDensity_ = db_->targetDensity();

  poissonSolver_ 
    = std::make_unique<PoissonSolver>(numBinX_, numBinY_);

  initForCUDAKernel();

  printf("[DensityGradient] Finish Initialization.\n");
}

DensityGradient::~DensityGradient()
{
  freeDeviceMemory();
}

void
DensityGradient::resetMacroDensityWeight()
{
	thrust::host_vector<float> h_cellDensityScale(numMovable_);

	int i = 0;
	for( auto& cell : db_->movableCells() )
		h_cellDensityScale[i++] = cell->densityScale();

	thrust::copy(h_cellDensityScale.begin(),
               h_cellDensityScale.end(),
               d_cellDensityScale_.begin());
}

void
DensityGradient::freeDeviceMemory()
{
}

} // namespace skyplace
