#include <cstdio>
#include <memory>
#include <chrono>
#include <cassert>

#include "CUDA_UTIL.h"
#include "NesterovOptimizer.h"

namespace skyplace 
{

__global__ void moveBackwardKernel(const int numCell,
                                   const float  stepLength, 
                                   const float  dieLx,
                                   const float  dieLy,
                                   const float  dieUx,
                                   const float  dieUy,
                                   const float* cellDx,
                                   const float* cellDy,
                                   const float* curPreX,
                                   const float* curPreY,
                                   const float* curPreTotalGradX,
                                   const float* curPreTotalGradY,
                                         float* prevPreX,
                                         float* prevPreY)
{
  // i := cellID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numCell)
  {
    // move Backward
    float prevCoordiX = 
      curPreX[i] - stepLength * curPreTotalGradX[i];
  
    float prevCoordiY = 
      curPreY[i] - stepLength * curPreTotalGradY[i];
  
    prevPreX[i] 
      = getXCoordiInsideLayoutDevice(prevCoordiX, cellDx[i], dieLx, dieUx);
    prevPreY[i] 
      = getYCoordiInsideLayoutDevice(prevCoordiY, cellDy[i], dieLy, dieUy);
  }
}

__global__ void moveForwardKernel(const int    numCell,
                                  const float  stepLength, 
                                  const float  stepLengthForPrediction, 
                                  const float  dieLx,
                                  const float  dieLy,
                                  const float  dieUx,
                                  const float  dieUy,
                                  const float* cellDx,
                                  const float* cellDy,
                                  const float* curCoordiX,
                                  const float* curCoordiY,
                                  const float* curPreCoordiX,
                                  const float* curPreCoordiY,
                                  const float* curPreTotalGradX,
                                  const float* curPreTotalGradY,
                                        float* nextCoordiX,
                                        float* nextCoordiY,
                                        float* nextPreCoordiX,
                                        float* nextPreCoordiY)
{
  // i := cellID
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < numCell)
  {
    float nextX = 
      curPreCoordiX[i] + stepLength * curPreTotalGradX[i];
  
    float nextY = 
      curPreCoordiY[i] + stepLength * curPreTotalGradY[i];
  
    float nextPreX = 
      nextX + stepLengthForPrediction * (nextX - curCoordiX[i]);
  
    float nextPreY = 
      nextY + stepLengthForPrediction * (nextY - curCoordiY[i]);
  
    nextCoordiX[i] 
      = getXCoordiInsideLayoutDevice(nextX, cellDx[i], dieLx, dieUx);
    nextCoordiY[i] 
      = getYCoordiInsideLayoutDevice(nextY, cellDy[i], dieLy, dieUy);

    nextPreCoordiX[i] 
      = getXCoordiInsideLayoutDevice(nextPreX, cellDx[i], dieLx, dieUx);
    nextPreCoordiY[i] 
      = getYCoordiInsideLayoutDevice(nextPreY, cellDy[i], dieLy, dieUy);
  }
}

NesterovOptimizer::NesterovOptimizer()
  : numCell_                    (0),
    numNet_                     (0),
    numPin_                     (0),

    dieLx_                      (0.0),
    dieLy_                      (0.0),
    dieUx_                      (0.0),
    dieUy_                      (0.0),

    wa_                         (0),
    hpwl_                       (0),
    prevHpwl_                   (0.0),
    overflow_                   (0.0),
    sumPenalty_                 (0.0),

    curA_                       (0),
    stepLength_                 (0),
    param_                      (nullptr),
    onlyGradMode_               (true   ),
    isDiverge_                  (false  ),
    db_                         (nullptr),
    painter_                    (nullptr),
    targetFunction_             (nullptr),

    d_ptr_cellWidth_            (nullptr),
    d_ptr_cellHeight_           (nullptr),

    d_ptr_prevPreX_             (nullptr),
    d_ptr_prevPreTotalGradX_    (nullptr),
    d_ptr_prevPreY_             (nullptr),
    d_ptr_prevPreTotalGradY_    (nullptr),
    d_ptr_curPreX_              (nullptr),
    d_ptr_curPreTotalGradX_     (nullptr),
    d_ptr_curPreY_              (nullptr),
    d_ptr_curPreTotalGradY_     (nullptr),
    d_ptr_nextPreX_             (nullptr),
    d_ptr_nextPreTotalGradX_    (nullptr),
    d_ptr_nextPreY_             (nullptr),
    d_ptr_nextPreTotalGradY_    (nullptr),
    d_ptr_curX_                 (nullptr),
    d_ptr_nextX_                (nullptr),
    d_ptr_curY_                 (nullptr),
    d_ptr_nextY_                (nullptr),

    initTime_                   (0.0),
    nesterovTime_               (0.0)
{}

NesterovOptimizer::NesterovOptimizer(std::shared_ptr<HyperParam>     param,
                                     std::shared_ptr<SkyPlaceDB>     db,
                                     std::shared_ptr<TargetFunction> func,
                                     std::shared_ptr<Painter>        painter) 
  : NesterovOptimizer()
{
  param_          = param;
  db_             = db;
  targetFunction_ = func;
  painter_        = painter;

  dieLx_ = db_->die()->lx();
  dieLy_ = db_->die()->ly();
  dieUx_ = db_->die()->ux();
  dieUy_ = db_->die()->uy();
}

NesterovOptimizer::~NesterovOptimizer()
{
  freeDeviceMemory();
}

void
NesterovOptimizer::initOptimizer()
{
  printf("[Nesterov] Optimizer Initialization\n");

  auto t1 = std::chrono::high_resolution_clock::now();

  param_->printHyperParameters();

  initForCUDAKernel();
  
  // Step #1. Compute Initial Gradient
  targetFunction_->getInitialGrad(d_ptr_curPreX_,
                                  d_ptr_curPreY_,
                                  d_ptr_curPreTotalGradX_,
                                  d_ptr_curPreTotalGradY_);

  // Step #2. Compute Initial Previous Coordinates
  // Since we don't have the previous predicted coordinates
  // at the first iteration, we have to make it up.
  moveBackward(param_->initOptCoef,
               d_ptr_curPreX_,
               d_ptr_curPreY_,
               d_ptr_curPreTotalGradX_,
               d_ptr_curPreTotalGradY_,
               d_ptr_prevPreX_,
               d_ptr_prevPreY_);
  // Compute the first Previous Predicted X / Y by going backward

  // Step #3. Update Pin Coordinates based on
  // Initial Previous Predicted Coordinates.
  // Compute the Previous Predicted Gradient 
  // based on the previous predicted Coordinate
  targetFunction_->updatePointAndGetGrad(d_prevPreX_,
                                         d_prevPreY_,
                                         d_prevPreTotalGradX_,
                                         d_prevPreTotalGradY_);

  // Step #4. Compute the initial Step Length
  stepLength_ = computeLipschitz(d_prevPreX_, 
                                 d_prevPreY_,
                                 d_prevPreTotalGradX_,
                                 d_prevPreTotalGradY_,
                                 d_curPreX_, 
                                 d_curPreY_, 
                                 d_curPreTotalGradX_,
                                 d_curPreTotalGradY_);

  targetFunction_->updateParameters(onlyGradMode_);
  prevHpwl_ = hpwl_;
  hpwl_     = targetFunction_->getHPWL();
  overflow_ = targetFunction_->getOverflow();

  if(!onlyGradMode_)
    sumPenalty_ = targetFunction_->getPenalty();

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> runtime = t2 - t1;

  initTime_ += runtime.count();

  printf("[Nesterov] Initial StepLength: %.1E\n", stepLength_);
  printf("[Nesterov] Optimizer Initialization Finished.\n");
}

void
NesterovOptimizer::diverge()
{
  isDiverge_ = true;
  printf("[Nesterov] Divergence Detected!\n");
  printf("[Nesterov] Terminate Placer...\n");
  exit(0);
}

Stat
NesterovOptimizer::startOptimize(bool plotMode)
{
  auto t1 = std::chrono::high_resolution_clock::now();

  initOptimizer();

  printf("[Nesterov] [StartGP] HPWL : %-6.3f  Overflow : %-3.3f\n",
                                  hpwl_ / 1e6, overflow_);

  int iter = 0;
  int backTrackIter = 0;
  curA_ = 1.0;

  if(plotMode)
    painter_->prepareForPlot();

  for(; iter < param_->maxOptIter; iter++)
  {
    float prevA = curA_;
    curA_ = 0.5 * (1.0 + std::sqrt(4.0 * prevA * prevA + 1.0));
    float coeff = (prevA - 1.0) / curA_;

    stepLength_ = backTracking(iter, coeff, backTrackIter);

    updateOneIteration(iter, backTrackIter);

    if(plotMode && (iter % 5 == 0))
    {
      opt2db();
      copyPotential2db();
      painter_->saveImage(iter, hpwl_, overflow_, false);
    }

    if(overflow_ <= param_->targetOverflow)
    {
      if(hpwl_ < prevHpwl_)
        continue;
      isDiverge_ = false;
      printf("[Nesterov] Convergence!\n");
      break;
    }
  }

  opt2db();

  // For Visualization
  copyPotential2db();
  copyDensityGrad2db();
  copyBinDensity2db();

  if(plotMode)
    painter_->saveImage(iter, hpwl_, overflow_, false); // no Filler

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> runtime = t2 - t1;

  nesterovTime_ = runtime.count();

  Stat finalStat 
    = {!isDiverge_, 
        hpwl_, 
        overflow_, 
        iter, 
        nesterovTime_, 
        targetFunction_->getWLGradTime(),
        targetFunction_->getDensityTime(),
        targetFunction_->getPoissonTime(),
        targetFunction_->getBinDenUpTime(),
        initTime_};

  return finalStat;  
}

float
NesterovOptimizer::backTracking(int iter, float coeff, int& backTrackIter)
{
  backTrackIter = 0;
  float newStepLength = param_->minStepLength;

  for(; backTrackIter < param_->maxBackTrackIter; backTrackIter++)
  {
    moveForward(stepLength_, 
                coeff,
                d_ptr_curX_,
                d_ptr_curY_,
                d_ptr_curPreX_,
                d_ptr_curPreY_,
                d_ptr_curPreTotalGradX_,
                d_ptr_curPreTotalGradY_,
                d_ptr_nextX_,
                d_ptr_nextY_,
                d_ptr_nextPreX_,
                d_ptr_nextPreY_);
  
    targetFunction_->updatePointAndGetGrad(d_nextPreX_,
                                           d_nextPreY_,
                                           d_nextPreTotalGradX_,
                                           d_nextPreTotalGradY_);

    newStepLength = computeLipschitz(d_curPreX_,
                                     d_curPreY_,
                                     d_curPreTotalGradX_,
                                     d_curPreTotalGradY_,
                                     d_nextPreX_,
                                     d_nextPreY_,
                                     d_nextPreTotalGradX_,
                                     d_nextPreTotalGradY_);
  
    if(newStepLength >= stepLength_ * 0.95)
      break;
    else if(newStepLength < param_->minStepLength) // minStepLength = 0.1
    {
      newStepLength = param_->minStepLength;
      break;
    }
  }
  return newStepLength;
}

void
NesterovOptimizer::moveForward(const float stepLength, 
                               const float stepLengthForPrediction,
                               const float* d_ptr_curCoordiX,
                               const float* d_ptr_curCoordiY,
                               const float* d_ptr_curPreCoordiX,
                               const float* d_ptr_curPreCoordiY,
                               const float* d_ptr_directionX,
                               const float* d_ptr_directionY,
                                     float* d_ptr_nextCoordiX,
                                     float* d_ptr_nextCoordiY,
                                     float* d_ptr_nextPreCoordiX,
                                     float* d_ptr_nextPreCoordiY)
{
  int numThread = 64;
  int numBlockCell = (numCell_ - 1 + numThread) / numThread;

  moveForwardKernel<<<numBlockCell, numThread>>>(numCell_, 
                                                 stepLength, 
                                                 stepLengthForPrediction,
                                                 dieLx_, 
                                                 dieLy_, 
                                                 dieUx_, 
                                                 dieUy_,
                                                 d_ptr_cellWidth_, 
                                                 d_ptr_cellHeight_,
                                                 d_ptr_curCoordiX, 
                                                 d_ptr_curCoordiY, 
                                                 d_ptr_curPreCoordiX, 
                                                 d_ptr_curPreCoordiY, 
                                                 d_ptr_directionX,
                                                 d_ptr_directionY,
                                                 d_ptr_nextCoordiX, 
                                                 d_ptr_nextCoordiY, 
                                                 d_ptr_nextPreCoordiX,
                                                 d_ptr_nextPreCoordiY);
}

void
NesterovOptimizer::moveBackward(const float  stepLength, 
                                const float* d_ptr_curCoordiX,
                                const float* d_ptr_curCoordiY,
                                const float* d_ptr_directionX,
                                const float* d_ptr_directionY,
                                      float* d_ptr_nextCoordiX,
                                      float* d_ptr_nextCoordiY)
{
  int numThread = 64;
  int numBlockCell = (numCell_ - 1 + numThread) / numThread;

  moveBackwardKernel<<<numBlockCell, numThread>>>(numCell_, 
                                                  stepLength, 
                                                  dieLx_, 
                                                  dieLy_, 
                                                  dieUx_, 
                                                  dieUy_,
                                                  d_ptr_cellWidth_, 
                                                  d_ptr_cellHeight_,
                                                  d_ptr_curCoordiX, 
                                                  d_ptr_curCoordiY, 
                                                  d_ptr_directionX,
                                                  d_ptr_directionY,
                                                  d_ptr_nextCoordiX,
                                                  d_ptr_nextCoordiY);
}

float
NesterovOptimizer::computeLipschitz(const thrust::device_vector<float>& d_curPreX, 
                                    const thrust::device_vector<float>& d_curPreY,
                                    const thrust::device_vector<float>& d_curPreTotalGradX,
                                    const thrust::device_vector<float>& d_curPreTotalGradY,
                                    const thrust::device_vector<float>& d_nextPreX, 
                                    const thrust::device_vector<float>& d_nextPreY, 
                                    const thrust::device_vector<float>& d_nextPreTotalGradX,
                                    const thrust::device_vector<float>& d_nextPreTotalGradY)
{
  float coordiDistance = 0;
  float gradDistance = 0;

  coordiDistance  = compute2NormSquare(d_curPreX, 
                                       d_nextPreX, 
                                       d_workSpaceForStepLength_);

  coordiDistance += compute2NormSquare(d_curPreY, 
                                       d_nextPreY, 
                                       d_workSpaceForStepLength_);

  coordiDistance = std::sqrt( coordiDistance 
                   / ( 2.0 * static_cast<float>(numCell_) ) );

  gradDistance  = compute2NormSquare(d_curPreTotalGradX, 
                                     d_nextPreTotalGradX, 
                                     d_workSpaceForStepLength_);

  gradDistance += compute2NormSquare(d_curPreTotalGradY, 
                                     d_nextPreTotalGradY, 
                                     d_workSpaceForStepLength_);

  gradDistance = std::sqrt( gradDistance 
                 / ( 2.0 * static_cast<float>(numCell_) ) );

  // printf("CoordiDist: %E GradDist: %E\n", coordiDistance, gradDistance); 

  return coordiDistance / gradDistance;
}

void
NesterovOptimizer::initForCUDAKernel()
{
  printf("[CUDA-Nesterov] Start Initialization.\n");

  numCell_ = db_->movableCells().size();
  numNet_  = db_->nets().size();
  numPin_  = db_->pins().size();

  h_cellCx_.resize(numCell_);
  h_cellCy_.resize(numCell_);

  std::vector<float> h_cellWidth_(numCell_);
  std::vector<float> h_cellHeight_(numCell_);
  std::vector<float> h_cellArea_(numCell_);

  std::vector<float> h_pinX_(numPin_);
  std::vector<float> h_pinY_(numPin_);

  // Not sure cellID is in the same order with
  // cellVector
  for(auto& cell : db_->movableCells())
  {
    int cellID = cell->id();

    h_cellCx_[cellID] = cell->cx();
    h_cellCy_[cellID] = cell->cy();

    h_cellWidth_[cellID]  = cell->dx();
    h_cellHeight_[cellID] = cell->dy();
    h_cellArea_[cellID]   = cell->area();
  }

  // Step #1. Resize Thrust Vector and get raw_pointer
  d_workSpaceForStepLength_.resize(numCell_);

  d_ptr_cellWidth_             = setThrustVector(numCell_, d_cellWidth_);
  d_ptr_cellHeight_            = setThrustVector(numCell_, d_cellHeight_);

  d_ptr_prevPreX_              = setThrustVector(numCell_, d_prevPreX_);
  d_ptr_prevPreTotalGradX_     = setThrustVector(numCell_, d_prevPreTotalGradX_);

  d_ptr_prevPreY_              = setThrustVector(numCell_, d_prevPreY_);
  d_ptr_prevPreTotalGradY_     = setThrustVector(numCell_, d_prevPreTotalGradY_);

  d_ptr_curPreX_               = setThrustVector(numCell_, d_curPreX_);
  d_ptr_curPreTotalGradX_      = setThrustVector(numCell_, d_curPreTotalGradX_);

  d_ptr_curPreY_               = setThrustVector(numCell_, d_curPreY_);
  d_ptr_curPreTotalGradY_      = setThrustVector(numCell_, d_curPreTotalGradY_);

  d_ptr_nextPreX_              = setThrustVector(numCell_, d_nextPreX_);
  d_ptr_nextPreTotalGradX_     = setThrustVector(numCell_, d_nextPreTotalGradX_);

  d_ptr_nextPreY_              = setThrustVector(numCell_, d_nextPreY_);
  d_ptr_nextPreTotalGradY_     = setThrustVector(numCell_, d_nextPreTotalGradY_);

  d_ptr_curX_                  = setThrustVector(numCell_, d_curX_);
  d_ptr_nextX_                 = setThrustVector(numCell_, d_nextX_);

  d_ptr_curY_                  = setThrustVector(numCell_, d_curY_);
  d_ptr_nextY_                 = setThrustVector(numCell_, d_nextY_);

  // Step #2. Synchronize Cell&Pin Cooridinates
  d_curPreX_  = h_cellCx_;
  d_curX_     = h_cellCx_;

  d_curPreY_  = h_cellCy_;
  d_curY_     = h_cellCy_;
  // We don't have to initialize Previous X / Y

  // Host -> Device
  d_cellWidth_  = h_cellWidth_;
  d_cellHeight_ = h_cellHeight_;

  printf("[CUDA-Nesterov] Finish Initialization.\n");
}

void
NesterovOptimizer::updateOneIteration(int iter, int backTrackIter)
{
  // Previous <= Current
  d_prevPreX_.swap(d_curPreX_);
  d_prevPreY_.swap(d_curPreY_);

  d_prevPreTotalGradX_.swap(d_curPreTotalGradX_);
  d_prevPreTotalGradY_.swap(d_curPreTotalGradY_);

  // Current <= Next
  d_curPreX_.swap(d_nextPreX_);
  d_curPreY_.swap(d_nextPreY_);

  d_curPreTotalGradX_.swap(d_nextPreTotalGradX_);
  d_curPreTotalGradY_.swap(d_nextPreTotalGradY_);

  d_curX_.swap(d_nextX_);
  d_curY_.swap(d_nextY_);

  // Re-Extract Raw Pointer
  d_ptr_prevPreX_              = thrust::raw_pointer_cast(&d_prevPreX_[0]);
  d_ptr_prevPreTotalGradX_     = thrust::raw_pointer_cast(&d_prevPreTotalGradX_[0]);

  d_ptr_prevPreY_              = thrust::raw_pointer_cast(&d_prevPreY_[0]);
  d_ptr_prevPreTotalGradY_     = thrust::raw_pointer_cast(&d_prevPreTotalGradY_[0]);

  d_ptr_curPreX_               = thrust::raw_pointer_cast(&d_curPreX_[0]);
  d_ptr_curPreTotalGradX_      = thrust::raw_pointer_cast(&d_curPreTotalGradX_[0]);

  d_ptr_curPreY_               = thrust::raw_pointer_cast(&d_curPreY_[0]);
  d_ptr_curPreTotalGradY_      = thrust::raw_pointer_cast(&d_curPreTotalGradY_[0]);

  d_ptr_nextPreX_              = thrust::raw_pointer_cast(&d_nextPreX_[0]);
  d_ptr_nextPreTotalGradX_     = thrust::raw_pointer_cast(&d_nextPreTotalGradX_[0]);

  d_ptr_nextPreY_              = thrust::raw_pointer_cast(&d_nextPreY_[0]);
  d_ptr_nextPreTotalGradY_     = thrust::raw_pointer_cast(&d_nextPreTotalGradY_[0]);

  d_ptr_curX_                  = thrust::raw_pointer_cast(&d_curX_[0]);
  d_ptr_nextX_                 = thrust::raw_pointer_cast(&d_nextX_[0]);

  d_ptr_curY_                  = thrust::raw_pointer_cast(&d_curY_[0]);
  d_ptr_nextY_                 = thrust::raw_pointer_cast(&d_nextY_[0]);

  targetFunction_->updateParameters(onlyGradMode_);
  prevHpwl_      = hpwl_;
  hpwl_          = targetFunction_->getHPWL();
  overflow_      = targetFunction_->getOverflow();
  macroOverflow_ = targetFunction_->getMacroOverflow();

  printf("[Nesterov] [Iter%04d] HPWL : %-6.3f  Overflow : %-3.3f\n", 
                          iter,     hpwl_ / 1e6 / db_->getDbu(),       overflow_); 

 // printf("[Nesterov] [Iter%04d] HPWL : %-6.3f  Overflow : %-3.3f  MacroOverflow : %-4.4f\n", 
  //                        iter,     hpwl_ / 1e6,       overflow_,     macroOverflow_); 

  //printf("[Nesterov] [Iter%03d] HPWL : %-6.3f  Overflow : %-3.3f BT : %02d  StepLength: %E\n", 
  //                        iter, hpwl_ / 1e6, overflow_, backTrackIter, stepLength_); 

  if(!onlyGradMode_)
  {
    wa_          = targetFunction_->getWA();
    sumPenalty_  = targetFunction_->getPenalty();
    float lambda = targetFunction_->getLambda();
    float functionValue = wa_ + lambda * sumPenalty_;
    printf("WA + lambda * Penalty = %E + %E * %E = %E\n", 
            wa_, lambda, sumPenalty_, functionValue);
  }
}

void
NesterovOptimizer::opt2db()
{
  thrust::copy(d_curX_.begin(), d_curX_.end(), h_cellCx_.begin());
  thrust::copy(d_curY_.begin(), d_curY_.end(), h_cellCy_.begin());

  int cIdx = 0;
  for(auto cell : db_->movableCells())
  {
    cell->setCenterLocation(h_cellCx_[cIdx], h_cellCy_[cIdx]);
    cIdx++;
  }
}

void
NesterovOptimizer::copyPotential2db()
{
  int numBin = static_cast<int>(db_->bins().size());

  float* temp = new float[numBin];

  CUDA_CHECK(
    cudaMemcpy(temp, 
    targetFunction_->density()->getDevicePotential(),
    numBin * sizeof(float),
    cudaMemcpyDeviceToHost));

  int binID = 0;
  for(auto& bin : db_->bins())
    bin->setElectroPotential(temp[binID++]);

  delete [] temp;
}

void
NesterovOptimizer::copyBinDensity2db()
{
  int numBin = static_cast<int>(db_->bins().size());

  float* temp = new float[numBin];

  CUDA_CHECK(
    cudaMemcpy(temp, 
    targetFunction_->density()->getDeviceBinDensity(),
    numBin * sizeof(float),
    cudaMemcpyDeviceToHost));

  int binID = 0;
  for(auto& bin : db_->bins())
    bin->setDensity(temp[binID++]);

  delete [] temp;
}

void
NesterovOptimizer::copyDensityGrad2db()
{
  db_->densityGradX().resize(numCell_);
  db_->densityGradY().resize(numCell_);

  thrust::copy(targetFunction_->densityGradX().begin(), 
               targetFunction_->densityGradX().end(), 
               db_->densityGradX().begin());

  thrust::copy(targetFunction_->densityGradY().begin(), 
               targetFunction_->densityGradY().end(), 
               db_->densityGradY().begin());
}

void
NesterovOptimizer::freeDeviceMemory()
{
}

}; // namespace skyplace
